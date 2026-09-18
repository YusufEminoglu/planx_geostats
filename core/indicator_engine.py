# -*- coding: utf-8 -*-
"""Indicator kriging: exceedance probabilities instead of a single surface.

Every other interpolator in this plugin answers one question - what is the value
here - and reports its uncertainty as a variance around that answer.  A variance
is a poor answer for the question a planner usually has, which is a threshold
question: how likely is this parcel to be above the flood level, below the
service standard, over the contamination limit.  Indicator kriging answers that
question directly, and it does so without assuming the field is symmetric or
even roughly gaussian, which is exactly what a variance assumes.

The method (Journel 1983; Goovaerts 1997, ch. 6) turns one continuous variable
into a family of 0/1 variables.  For each threshold ``z_k`` the indicator is

    I(s; z_k) = 1 if Z(s) <= z_k, else 0

so its expectation is the conditional cumulative probability ``F(z_k)`` at that
location, and kriging the indicator estimates it.  Nothing about that step
requires a distributional assumption; the price is that the fit is repeated once
per threshold, and that each threshold's indicator carries its own variogram.

Two facts this module is built around, because both are easy to get wrong and
neither is visible in an output surface:

* **The raw estimates are not a valid distribution.**  Indicator kriging
  estimates each threshold separately, so nothing enforces that the fitted
  curve rises with the threshold.  It routinely does not: at a cell between two
  observations the estimate for a high threshold can come out below the estimate
  for a low one.  Those are the *order relations*, and a curve that violates
  them is not a probability distribution at all.  :func:`correct_order_relations`
  repairs them by the standard GSLIB averaging rule and reports how much repair
  was needed, because the size of the repair is a diagnostic: a large one means
  the thresholds are too close together for the sampling to separate them.

* **The E-type estimate is not an exact interpolator**, even though indicator
  kriging is.  At an observed location the fitted curve is a perfect step -
  zero below the datum, one at and above it - so it says only which class the
  datum falls in, and the expected value of the discretised distribution is that
  class's mean rather than the datum itself.  The gap is bounded by the class
  width and vanishes as the thresholds are packed more densely, but it does not
  go away, and a report that presents the E-type surface as reproducing the
  observations would be wrong.  :func:`etype_from_probabilities` returns the
  class means alongside the surface so the report can say this in numbers.

numpy-only, matching the rule the rest of ``core`` follows.  No QGIS import:
this module is importable and testable without a QGIS runtime.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .geostat_engines import (
    MODELS,
    default_max_lag,
    empirical_variogram,
    fit_variogram,
    krige_chunked,
)
from .interpolation_engines import cv_protocol_for

logger = logging.getLogger("PlanX GeoStats Lab")

#: Families an indicator variogram may be fitted with.  The linear model is
#: admitted here although Empirical Bayesian Kriging refuses it: EBK
#: *simulates* from the fitted model and an unbounded variogram has no
#: covariance to simulate from, whereas indicator kriging only ever solves the
#: ordinary-kriging system, which the linear model satisfies perfectly well.
IK_MODELS = MODELS

#: Below this many observations there is not enough of either class to fit an
#: indicator variogram: a threshold near the tail of the data leaves a handful
#: of ones, and their semivariance is noise.
IK_MIN_POINTS = 8

#: Thresholds used when the caller names none.  Five reproduces the conventional
#: five-class map and keeps the number of probability bands small enough to be
#: read, while still resolving the middle of the distribution.
DEFAULT_THRESHOLD_COUNT = 5

#: Leave-one-out for this method costs one variogram refit per observation per
#: threshold, so the affordable count is far lower than for the deterministic
#: interpolators, where a held-out point costs one kriging solve and no fit.
#: Above this, k-fold runs instead and the report names the protocol, because
#: the two are not comparable - a fold fits on a fraction of the data, so its
#: Brier score is systematically worse.
IK_LOO_LIMIT = 200

#: Folds used when leave-one-out is not affordable, matching the convention the
#: deterministic interpolators already document.
IK_DEFAULT_FOLDS = 5

#: Cell sizes for the declustering check, as fractions of the longer side of
#: the observations' bounding box.  The usual advice is to sweep the cell size
#: and read the plateau rather than to nominate one, because the answer moves
#: with the choice and a single size cannot show that it does.  A single
#: observation once covered the whole domain is dropped from the sweep because
#: its weight is then 1 and the result is the plain mean again.
DECLUSTER_CELL_FRACTIONS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75)

#: Thresholds closer together than this fraction of the field's range are
#: reported as unresolvable rather than silently fitted: the two indicators
#: differ on a handful of observations, so their variograms are the same curve
#: fitted to noise and the order-relation repair between them carries no
#: information.
MIN_THRESHOLD_SEPARATION = 1e-6

#: Observation cap.  An indicator variogram is an ordinary kriging system, so
#: the cap is the same one ordinary kriging uses - but the cost is that cap's
#: cost multiplied by the threshold count, which is what the work model below
#: is for.
IK_SOFT_LIMIT = 2000

#: The cost of this method is the number of output cells multiplied by the
#: number of observations multiplied by the number of thresholds.  Measured on
#: the engine's synthetic fields at 20 000 to 160 000 cells, 200 to 600
#: observations and 5 to 10 thresholds, that product runs at 1.4 to 1.7 million
#: units per second; the figure below is the slow end of that range, so an
#: estimate built from it over-states the wait rather than under-stating it.
IK_WORK_PER_SECOND = 1.4e7

#: Roughly twenty seconds, and roughly three minutes.  Both are the same shape
#: as the limits Empirical Bayesian Kriging applies, so a user who has met one
#: of these tools already knows what the other is telling them.
IK_WORK_SOFT_LIMIT = 2.8e8
IK_WORK_HARD_LIMIT = 2.5e9


def estimate_work(n_targets, n_points, n_thresholds):
    """The cost model behind the two limits above, in the same units."""
    return (float(n_targets) * float(n_points) * float(n_thresholds))


def check_workload(n_targets, n_points, n_thresholds,
                   where="Indicator kriging"):
    """Return a warning string when a run is slow, raise when it is hopeless.

    All three factors are named in the refusal with their current values,
    because all three are things the caller can change and "too slow" is not.
    """
    work = estimate_work(n_targets, n_points, n_thresholds)
    seconds = work / IK_WORK_PER_SECOND
    if work > IK_WORK_HARD_LIMIT:
        raise ValueError(
            f"{where} would need about {seconds:.0f} s for "
            f"{int(n_targets):,} target cells, {int(n_points):,} observations "
            f"and {int(n_thresholds)} thresholds. Reduce any of the three: the "
            "cost is their product, so halving the threshold count halves it. A "
            "larger cell size, fewer thresholds and fewer observations are the "
            "three ways down."
        )
    if work > IK_WORK_SOFT_LIMIT:
        return (f"{int(n_targets):,} cells x {int(n_points):,} observations x "
                f"{int(n_thresholds)} thresholds is roughly {seconds:.0f} s of "
                "computation, because every threshold is a separate kriging "
                "system solved over the whole grid.")
    return ""


def parse_thresholds(text, values=None, count=DEFAULT_THRESHOLD_COUNT):
    """Resolve the thresholds a run will use, in ascending order.

    ``text`` is the analyst's own list, comma or semicolon separated, and takes
    precedence: a regulatory limit is a number someone else chose, and replacing
    it with a quantile of the local data would answer a different question.
    With no list, ``count`` quantiles of the observed values are used, which is
    the honest default - it puts the thresholds where the data can actually
    separate them - but the report says plainly that they are quantiles of this
    sample and carry no external meaning.

    Returns ``(thresholds, source)`` where ``source`` is ``"given"`` or
    ``"quantiles"``.  Duplicates are collapsed: two identical thresholds produce
    two identical indicators and a redundant probability band.
    """
    if text is not None and str(text).strip():
        pieces = str(text).replace(";", ",").split(",")
        parsed = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            try:
                parsed.append(float(piece))
            except ValueError as exc:
                raise ValueError(
                    f"'{piece}' is not a number, so the threshold list "
                    f"'{text}' cannot be read. Give the thresholds as numbers "
                    "separated by commas, for example 10, 25, 50."
                ) from exc
        if not parsed:
            raise ValueError(
                f"No thresholds could be read from '{text}'. Give at least one "
                "number, or leave the field empty to use quantiles of the data."
            )
        thresholds = np.unique(np.asarray(parsed, dtype=float))
        if len(thresholds) < 1:
            raise ValueError("At least one threshold is needed.")
        return thresholds, "given"

    if values is None:
        raise ValueError(
            "Thresholds were neither given nor derived: the observed values are "
            "needed to place them at quantiles."
        )
    z = np.asarray(values, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        raise ValueError("There are no finite values to take quantiles from.")
    k = max(1, min(int(count), len(z)))
    quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]
    thresholds = np.unique(np.quantile(z, quantiles))
    if len(thresholds) == 0:
        raise ValueError(
            "The observed values are all identical, so every quantile is the "
            "same number and no threshold can separate them."
        )
    return thresholds, "quantiles"


def indicator_vector(values, threshold):
    """The 0/1 indicator of ``values <= threshold``, as floats.

    Floats rather than booleans because every consumer of this is an
    arithmetic one, and a bool array silently becomes the wrong dtype at the
    first ``@``.
    """
    z = np.asarray(values, dtype=float)
    return (z <= float(threshold)).astype(float)


def declustered_frequencies(coords, values, thresholds,
                            fractions=DECLUSTER_CELL_FRACTIONS):
    """Frequency of ``values <= t`` under cell-declustered weights.

    The fraction of the *observations* at or below a threshold and the fraction
    of the *mapped area* at or below it are two different numbers whenever the
    sampling is clustered, and a cluster sample is the normal case: observations
    follow roads, boreholes and survey lines, and the map covers everything
    between them.  Ordinary kriging's far field is the declustered mean of the
    data - that is the classical result, and it is why kriging is preferred to a
    distance-weighted average - so the surface will sit near the declustered
    frequency, not near the sample frequency.  Reported side by side, that is a
    check the reader can complete; reported one at a time, a surface averaging
    0.86 against a sample frequency of 0.75 reads as a defect when it is the
    method working.

    The weights are the ordinary cell-declustering ones: divide the domain into
    square cells of one size, weight each observation by the reciprocal of the
    number of observations sharing its cell, and normalize.  One observation
    alone in a cell therefore counts for as much as a whole crowded cell.

    ``fractions`` gives the cell sizes to sweep, as fractions of the longer side
    of the bounding box.  The answer moves with the cell size, so the median of
    the sweep is returned as the headline and the extremes as the uncertainty;
    a swept result that is not stable is itself worth reporting rather than
    hiding behind one arbitrary size.

    Returns a dict with ``cell_sizes`` (map units), ``frequencies`` (one row per
    cell size, one column per threshold), ``median``, ``low`` and ``high`` (one
    value per threshold), and ``counts`` (the largest number of observations
    sharing a cell at each size), which is how clustered the sample is.
    """
    c = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    finite = np.isfinite(z) & np.isfinite(c).all(axis=1)
    c, z = c[finite], z[finite]
    if len(z) == 0:
        raise ValueError("There are no finite observations to decluster.")
    if len(t) == 0:
        raise ValueError("At least one threshold is needed to decluster.")

    indicators = np.column_stack([indicator_vector(z, q) for q in t])
    lo = c.min(axis=0)
    span = float(np.max(c.max(axis=0) - lo))
    if not np.isfinite(span) or span <= 0.0:
        # Every observation shares a location, so no cell size can separate
        # them and the declustered frequency is the plain one.
        plain = indicators.mean(axis=0)
        return {
            "cell_sizes": np.zeros(0, dtype=float),
            "frequencies": np.zeros((0, len(t)), dtype=float),
            "median": plain,
            "low": plain,
            "high": plain,
            "counts": np.zeros(0, dtype=int),
        }

    rows = []
    cell_sizes = []
    counts = []
    for fraction in fractions:
        size = float(fraction) * span
        if size <= 0.0:
            continue
        # A ragged grid keyed by cell index: two observations share a cell
        # exactly when they share both indices, and the offset keeps the two
        # axes from colliding in one integer.
        ix = np.floor((c[:, 0] - lo[0]) / size).astype(np.int64)
        iy = np.floor((c[:, 1] - lo[1]) / size).astype(np.int64)
        _unique, inverse, per_cell = np.unique(
            ix * np.int64(1_000_003) + iy, return_inverse=True,
            return_counts=True,
        )
        weight = 1.0 / per_cell[inverse]
        weight = weight / weight.sum()
        rows.append(weight @ indicators)
        cell_sizes.append(size)
        counts.append(int(per_cell.max()))

    frequencies = np.vstack(rows)
    return {
        "cell_sizes": np.asarray(cell_sizes, dtype=float),
        "frequencies": frequencies,
        "median": np.median(frequencies, axis=0),
        "low": frequencies.min(axis=0),
        "high": frequencies.max(axis=0),
        "counts": np.asarray(counts, dtype=int),
    }


def class_means(values, thresholds):
    """Mean of the observations falling in each class, and the class edges.

    The classes are the intervals the thresholds cut the axis into: below the
    first threshold, between each adjacent pair, and above the last.  Their
    means are what the E-type estimate multiplies the class probabilities by,
    so an empty class has no mean to contribute.  Rather than let a zero stand
    in - which would drag the surface towards zero exactly where the data is
    thin - an empty class falls back to the midpoint of its own interval, and
    the count of substitutions is returned so the report can disclose it.

    Returns ``(means, edges, empty_classes)``: one mean per class, the
    ``T + 1`` interval boundaries, and the indices of the classes that had no
    observations.
    """
    z = np.asarray(values, dtype=float)
    z = z[np.isfinite(z)]
    t = np.asarray(thresholds, dtype=float)
    if len(t) == 0:
        raise ValueError("At least one threshold is needed to define classes.")
    if len(z) == 0:
        raise ValueError("There are no finite values to classify.")

    edges = np.concatenate(([-np.inf], t, [np.inf]))
    means = np.zeros(len(edges) - 1, dtype=float)
    empty = []
    for k in range(len(means)):
        low, high = float(edges[k]), float(edges[k + 1])
        if k == 0:
            member = z <= high
        elif k == len(means) - 1:
            member = z > low
        else:
            member = (z > low) & (z <= high)
        if member.any():
            means[k] = float(z[member].mean())
        else:
            empty.append(k)
            means[k] = _interval_midpoint(low, high, t, k)
    return means, edges, empty


def _interval_midpoint(low, high, thresholds, index):
    """A stand-in value for a class the sample left empty.

    An interior class is bounded on both sides and its midpoint is the obvious
    choice.  The two tail classes are not, so one class width is extrapolated
    outwards from the nearest threshold - the same gap the neighbouring
    threshold implies - rather than reaching for the field's observed minimum,
    which belongs to a different class and would flatten the tail it stands for.
    """
    t = np.asarray(thresholds, dtype=float)
    if np.isfinite(low) and np.isfinite(high):
        return 0.5 * (low + high)
    if np.isfinite(high):
        width = (t[1] - t[0]) if len(t) > 1 else max(abs(high), 1.0)
        return float(high) - 0.5 * float(width)
    width = (t[-1] - t[-2]) if len(t) > 1 else max(abs(low), 1.0)
    return float(low) + 0.5 * float(width)


def fit_indicator_model(coords, indicators, model="auto", alpha=1.5,
                        n_bins=15, max_lag=None):
    """Fit one semivariogram to one indicator vector.

    ``model="auto"`` tries every family in :data:`IK_MODELS` and keeps the one
    with the lowest Cressie-weighted cost, which is the same rule Empirical
    Bayesian Kriging applies per subset - so the two tools agree on what "best
    fitting" means even though they use the fit differently.

    Returns ``(fit, lag, semivariance, npairs)``.  The binned cloud is returned
    with the fit because the report draws it, and recomputing it there would
    mean a second O(n^2) pairing per threshold.
    """
    c = np.asarray(coords, dtype=float)
    ind = np.asarray(indicators, dtype=float)
    lag, semivariance, npairs = empirical_variogram(
        c, ind, n_bins=int(n_bins), max_lag=max_lag
    )[:3]
    if len(lag) < 4:
        raise ValueError(
            f"Only {len(lag)} populated distance bin(s) are available for an "
            "indicator variogram, and 4 are needed to fit one. Increase the bin "
            "count, raise the maximum lag distance, or use fewer thresholds - "
            "a threshold close to the tail of the data leaves too few "
            "observations on one side to describe any structure."
        )

    if model != "auto":
        if model not in IK_MODELS:
            raise ValueError(
                f"Unknown semivariogram model for indicator kriging: '{model}'. "
                f"Supported: {', '.join(IK_MODELS)}, or 'auto'."
            )
        fit = fit_variogram(lag, semivariance, npairs, model, alpha)
        return fit, lag, semivariance, npairs

    best = None
    for family in IK_MODELS:
        try:
            candidate = fit_variogram(lag, semivariance, npairs, family, alpha)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate
    if best is None:
        raise ValueError(
            "No semivariogram family could be fitted to this indicator. The "
            "indicator may be almost constant, which leaves nothing to fit - "
            "a threshold outside the range of the data produces exactly that."
        )
    return best, lag, semivariance, npairs


def indicator_is_constant(indicators):
    """True when the indicator takes a single value at every observation.

    A threshold above the largest observation makes every indicator 1; one at
    or below the smallest makes every indicator 0.  Both are ordinary questions
    - "how likely is this to exceed a level the sample never reached" is the
    whole point of asking about extremes - and both have an answer that needs
    no kriging at all, so the fitting this module would otherwise do has no
    information to work with.  Detected here rather than left to fail inside
    the fitter, which can only report that the semivariogram is flat without
    saying why.
    """
    ind = np.asarray(indicators, dtype=float)
    if len(ind) == 0:
        return True
    return bool(np.all(ind == ind[0]))


def exact_indicator_curve(indicators, n_targets):
    """The probability curve for an indicator that never varies.

    Every cell's probability is the indicator's own value, and the variance is
    zero: the threshold is outside the observed range, so the question "did
    this exceed it" has the same answer at every location and no estimation is
    involved.  Reporting a surface of zeros is the correct answer here, not a
    degenerate one, and the report says the threshold lay outside the data
    range so a reader is not misled into thinking a fit happened.

    Returns ``(estimate, variance)``, both arrays of ``n_targets``.
    """
    ind = np.asarray(indicators, dtype=float)
    if len(ind) == 0:
        raise ValueError("An indicator curve needs at least one observation.")
    value = float(ind[0])
    count = int(n_targets)
    return (np.full(count, value, dtype=float),
            np.zeros(count, dtype=float))


def indicator_krige(coords, indicators, targets, fit, progress=None):
    """Krige one indicator to the targets under one fitted model.

    Ordinary kriging, no drift.  The indicator is 0/1 and its mean is not known
    in advance - that mean is the quantity being estimated - so the unknown
    local mean that ordinary kriging absorbs is the right assumption and a
    simple-kriging system would have to be handed a number the analyst does not
    have.

    Targets are solved in chunks, so a raster-sized grid costs a bounded
    allocation per threshold rather than ``(n + 1) * len(targets)`` doubles -
    which at 160 000 cells and 300 observations would be several hundred
    megabytes *per threshold*, and this routine is called once per threshold.
    Chunking moves the last bits by about 1e-14 relative, far below what a
    Float32 raster can represent; with a single chunk the two are bit-for-bit
    equal, which is why the engine's exactness checks are unaffected.

    Returns ``(estimate, variance, overshoot)``.  ``overshoot`` counts the
    targets whose raw estimate fell outside [0, 1]: a probability estimate
    outside its own range is not a probability, and the count is reported
    before clipping rather than quietly removed.
    """
    t = np.atleast_2d(np.asarray(targets, dtype=float))
    if len(t) == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, 0
    est, var = krige_chunked(
        coords, indicators, t, fit["model"], fit["nugget"], fit["psill"],
        fit["range"], drift="none", alpha=fit.get("alpha", 1.5),
        progress=progress,
    )
    overshoot = int(np.count_nonzero((est < 0.0) | (est > 1.0)))
    return np.clip(est, 0.0, 1.0), np.maximum(var, 0.0), overshoot


def correct_order_relations(probabilities):
    """Repair the fitted curves so each is a valid distribution.

    ``probabilities`` is ``(T, M)``: one row per threshold, ascending, one
    column per target.  The repair is the standard one from GSLIB's ``order``
    routine and from Goovaerts (1997, p. 480): wherever an adjacent pair is
    descending, replace both by their average, and keep merging until the curve
    rises everywhere.  Averaging rather than clamping is what keeps the
    correction symmetric - clamping to the lower value would systematically
    bias every corrected curve downwards, and the bias would land hardest
    exactly where the estimates disagree most, which is between the
    observations, where the map is read.

    The merging is done as a pool-adjacent-violators pass, which computes the
    same answer as iterating the pairwise average to convergence - the merge
    step is that average - but in one pass and exactly, rather than
    geometrically approaching it.  That is not a matter of taste.  Where the
    iteration does converge the two agree to the last bit - one sweep on
    isolated violations, fifty-five on a cascading descent - but it does not
    always converge.  Three descending thresholds settle in fifty-five sweeps
    and four in a hundred and seven; five, and a three-threshold descent that
    lands on a flat tail, fail to settle in a hundred thousand.  Both are
    ordinary shapes for a dense axis over a region the sampling cannot resolve,
    and the cost grows with the axis rather than shrinking.

    Returns ``(corrected, report)``.  The report carries ``cells`` (targets
    needing any repair), ``clipped`` (estimates outside [0, 1] before the
    repair), ``max_block`` (the largest number of adjacent thresholds any single
    target had pooled into one probability) and ``max_adjustment`` (the largest
    change the repair made to any estimate).  Those four numbers are the
    diagnostic: a large ``max_block`` or ``max_adjustment`` means the thresholds
    are too close together for the sampling to tell them apart, and the curve is
    really reporting one probability rather than several.

    ``clipped`` counts what the caller handed in, so it is only meaningful when
    the caller passes raw kriging output.  :func:`indicator_krige` clips its own
    estimates before returning and reports its own count, so the indicator
    kriging tool hands this function arrays that are already inside [0, 1] and
    reads ``clipped`` there as zero for any data, which is a fact about the
    caller and not about the field.  A caller in that position should count
    clipping where it happens, not here.
    """
    raw = np.atleast_2d(np.asarray(probabilities, dtype=float))
    if raw.size == 0:
        empty = np.zeros_like(raw)
        return empty, {"cells": 0, "clipped": 0, "max_block": 0,
                       "max_adjustment": 0.0, "thresholds": int(raw.shape[0])}
    if raw.ndim != 2:
        raise ValueError(f"Expected a (thresholds, targets) array; got {raw.shape}.")

    clipped = int(np.count_nonzero((raw < 0.0) | (raw > 1.0)))
    if not np.isfinite(raw).all():
        raise ValueError(
            "The probability estimates are not all finite, so the "
            "order-relation repair cannot be applied. Non-finite kriging "
            "estimates come from duplicate observation locations carrying "
            "different values, which makes the kriging system singular. "
            "Remove the duplicates and retry."
        )
    out = np.clip(raw, 0.0, 1.0)
    n_thresholds, n_cells = out.shape
    if n_thresholds == 1:
        return out, {"cells": 0, "clipped": clipped, "max_block": 0,
                     "max_adjustment": 0.0, "thresholds": 1}

    # Pool adjacent violators.  The pairwise-averaging rule this replaced and
    # this one have the same fixed point - the merge step *is* that averaging -
    # but the iteration only walks towards it, geometrically, and on a long
    # threshold axis the walk outlasts any bound worth setting: a
    # fifteen-threshold curve that starts badly ordered needs more than ten
    # thousand sweeps to settle at double precision.  The harness found that by
    # exceeding exactly that bound.  Resolving each cell's blocks directly
    # reaches the same answer in one pass, and is exactly monotone rather than
    # monotone to a tolerance.
    #
    # Only the cells that violate something are walked.  On a real field most
    # curves are already ordered, and the ones that are not are the interesting
    # fraction the report counts.
    monotone = out.copy()
    needs = (out[:-1, :] > out[1:, :]).any(axis=0)
    block_sizes = np.zeros(n_cells, dtype=int)
    for cell in np.flatnonzero(needs):
        means: list[float] = []
        sizes: list[int] = []
        for value in out[:, cell]:
            mean = float(value)
            weight = 1
            while means and means[-1] > mean:
                pooled_mean = means.pop()
                pooled_weight = sizes.pop()
                mean = ((pooled_mean * pooled_weight + mean * weight)
                        / (pooled_weight + weight))
                weight = pooled_weight + weight
            means.append(mean)
            sizes.append(weight)
        at = 0
        for mean, weight in zip(means, sizes):
            monotone[at:at + weight, cell] = mean
            at += weight
        block_sizes[cell] = max(sizes)

    return monotone, {
        "cells": int(np.count_nonzero(needs)),
        "clipped": clipped,
        "max_block": int(block_sizes.max()) if n_cells else 0,
        "max_adjustment": float(np.max(np.abs(monotone - out))) if out.size else 0.0,
        "thresholds": int(n_thresholds),
    }


def exceedance_probabilities(corrected):
    """``P(Z > z_k)`` from the corrected ``P(Z <= z_k)``, still ``(T, M)``.

    Descending in the threshold, because it is one minus an ascending curve.
    Returned rather than computed by the caller so the direction convention is
    stated once: the bands the tool writes are exceedance, not cumulative, and
    a reader who assumes the other direction reads every map backwards.
    """
    f = np.atleast_2d(np.asarray(corrected, dtype=float))
    return np.clip(1.0 - f, 0.0, 1.0)


def etype_from_probabilities(corrected, means):
    """The expected value under the fitted distribution, and its variance.

    ``E[Z] = sum_k m_k * p_k`` where ``p_0 = F(z_1)``, ``p_k = F(z_k+1) -
    F(z_k)`` and ``p_T = 1 - F(z_T)``, and ``m_k`` are the class means.  The
    variance is the same sum over squares, ``sum_k m_k^2 p_k - E[Z]^2``.

    Neither is the kriging variance.  This variance describes the spread of the
    discretised distribution at a cell - how wide the class intervals are that
    the cell's probability mass sits in - and says nothing about how well the
    thresholds themselves were estimated.  The report carries both and names
    them differently, because presenting this one as a prediction variance
    would understate the real uncertainty by the whole of the indicator
    kriging variance.
    """
    f = np.atleast_2d(np.asarray(corrected, dtype=float))
    m = np.asarray(means, dtype=float)
    n_classes = f.shape[0] + 1
    if len(m) != n_classes:
        raise ValueError(
            f"Expected {n_classes} class means for {f.shape[0]} threshold(s); "
            f"got {len(m)}."
        )

    probabilities = np.empty((n_classes, f.shape[1]), dtype=float)
    probabilities[0, :] = f[0, :]
    if n_classes > 2:
        probabilities[1:-1, :] = f[1:, :] - f[:-1, :]
    probabilities[-1, :] = 1.0 - f[-1, :]
    # The corrected curve is monotone, so every class probability is already
    # non-negative; the arithmetic is the same either way, so the guard is an
    # assertion about the correction rather than a second repair.
    probabilities = np.maximum(probabilities, 0.0)

    mean = m @ probabilities
    second = (m ** 2) @ probabilities
    variance = np.maximum(second - mean ** 2, 0.0)
    return mean, variance, probabilities


def indicator_cross_validation(coords, values, thresholds, model="auto",
                               alpha=1.5, n_bins=15, max_lag=None,
                               protocol=None, folds=IK_DEFAULT_FOLDS, seed=0,
                               progress=None):
    """Score the fitted exceedance probabilities against held-out observations.

    The metric is the Brier score, ``mean((p_i - o_i)^2)`` over the held-out
    observations, where ``p_i`` is the kriged probability of exceeding the
    threshold and ``o_i`` is whether the observation actually did.  It is the
    right score for a probability forecast and the wrong one is RMSE: the
    quantity being predicted is a probability, so a squared error in
    probability units is what a user is exposed to, and an error in value units
    would not be comparable between thresholds.

    Each fold refits the indicator variogram, because the variogram is
    estimated from the same observations the prediction uses and a fold that
    borrowed it would be scoring the fit as well as the prediction.  That is
    why this is the most expensive validation in the plugin and why it runs on
    k-fold rather than leave-one-out above a modest point count.

    Returns a dict with a per-threshold list and an aggregate.  ``observed_freq``
    beside ``mean_probability`` is the calibration check: on a well-fitted field
    the average predicted probability of being at or below a threshold should
    land close to the fraction of held-out observations that were - both are
    computed from the same indicator, which is 1 when the value is at or below
    the threshold.  Note what that check can and cannot see.  Every score in it
    is taken at an observation, and indicator kriging is an exact interpolator,
    so the held-out curve is pinned to the data wherever it is being scored:
    a surface can pass this check and still be far from the mapped area's own
    frequency, because those are two different quantities when the sampling is
    clustered - see :func:`declustered_frequencies`.  ``n_dropped`` is the
    number of held-out scores the run attempted and could not produce, summed
    over every threshold that was estimated - each threshold drops its own
    points, so it is not a count of observations.

    ``held_out_predicted`` and ``held_out_observed`` are those same scores
    pooled across the thresholds that were scored, in threshold order, and are
    what :func:`calibration_table` bins.  A per-threshold mean cannot show a
    curve that is well scaled on average and over-confident in the middle of
    its range; binning the pooled scores can.
    """
    c = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    n = len(z)
    if n < IK_MIN_POINTS:
        raise ValueError(
            f"Indicator cross-validation needs at least {IK_MIN_POINTS} "
            f"observations; got {n}."
        )
    if protocol is None:
        protocol, resolved_folds = cv_protocol_for(n, IK_LOO_LIMIT)
        if protocol == "loo":
            folds = resolved_folds

    if protocol == "loo":
        groups = [np.array([i]) for i in range(n)]
    elif protocol == "kfold":
        k = max(2, min(int(folds), n))
        order = np.random.default_rng(int(seed)).permutation(n)
        groups = [order[i::k] for i in range(k)]
    else:
        raise ValueError(
            f"Unknown cross-validation protocol '{protocol}'. Expected 'loo' "
            "or 'kfold'."
        )

    total = len(groups) * len(t)
    done = 0
    per_threshold = []
    # Held-out predictions pooled across thresholds, so the caller can bin them
    # by predicted probability and check calibration, which a per-threshold
    # mean cannot show: a set of probabilities can score well overall and still
    # be systematically too confident in the middle of its range.
    held_out_predicted = []
    held_out_observed = []
    for k, threshold in enumerate(t):
        ind = indicator_vector(z, threshold)
        if indicator_is_constant(ind):
            # Every observation sits on the same side of this threshold, so the
            # probability is answered by the threshold's own position rather
            # than by any model, and a cross-validated forecast would score a
            # perfect zero without having predicted anything.  Counting that
            # zero in the average would lift the Brier skill score of a run
            # whose other thresholds are genuinely hard, so it is reported
            # separately and left out of the aggregate.
            per_threshold.append({
                "threshold": float(threshold),
                "n": 0,
                "brier": float("nan"),
                "observed_freq": float(ind.mean()),
                "mean_probability": float(ind.mean()),
                "exact": True,
                "reason": ("every observation is at or below this threshold"
                           if ind[0] > 0.5 else
                           "every observation is above this threshold"),
            })
            continue
        predicted = np.full(n, np.nan, dtype=float)
        for test in groups:
            train = np.ones(n, dtype=bool)
            train[test] = False
            try:
                fit, *_ = fit_indicator_model(
                    c[train], ind[train], model, alpha, n_bins, max_lag
                )
            except ValueError:
                # A fold whose training half has only one indicator class has
                # no variogram to fit.  Its held-out points stay NaN and are
                # counted as dropped rather than being scored against a
                # meaningless curve.
                done += 1
                continue
            est, _var, _over = indicator_krige(c[train], ind[train], c[test], fit)
            predicted[test] = est
            done += 1
            if progress is not None and progress(done, total) is False:
                raise RuntimeError("Indicator cross-validation cancelled.")

        usable = np.isfinite(predicted)
        if not usable.any():
            per_threshold.append({
                "threshold": float(threshold),
                "n": 0,
                "brier": float("nan"),
                "observed_freq": float((ind > 0.5).mean()),
                "mean_probability": float("nan"),
                "exact": False,
                "reason": "no held-out observation could be scored",
            })
            continue
        observed = ind[usable]
        held_out_predicted.append(predicted[usable])
        held_out_observed.append(observed)
        brier = float(np.mean((predicted[usable] - observed) ** 2))
        per_threshold.append({
            "threshold": float(threshold),
            "n": int(usable.sum()),
            "brier": brier,
            "observed_freq": float(observed.mean()),
            "mean_probability": float(predicted[usable].mean()),
            "max_probability": float(predicted[usable].max()),
            "min_probability": float(predicted[usable].min()),
            "exact": False,
            "reason": "",
        })

    exact_rows = [row for row in per_threshold if row["exact"]]
    estimable = [row for row in per_threshold if not row["exact"]]
    scored = [row for row in estimable if row["n"] > 0]
    # How many held-out scores the run set out to produce and did not.  Counted
    # across the thresholds, because each one drops its own points: a fold whose
    # training half carries a single indicator class cannot be fitted, and that
    # happens at one threshold and not its neighbour.  Comparing a sum over
    # thresholds against a single threshold's observation count - which is what
    # this did - reports a negative number the moment more than one threshold is
    # scored, so the count is taken against the scores actually attempted.
    dropped = len(estimable) * n - sum(row["n"] for row in estimable)
    mean_brier = float(np.mean([row["brier"] for row in scored])) if scored else float("nan")
    # The reference forecast a Brier score must beat: always predict the
    # threshold's own observed frequency, ignoring location entirely.  A
    # method that cannot beat that has learned nothing spatial.
    reference = float(np.mean([
        row["observed_freq"] * (1.0 - row["observed_freq"]) for row in scored
    ])) if scored else float("nan")
    skill = (1.0 - mean_brier / reference
             if reference and np.isfinite(reference) and reference > 0 else float("nan"))

    return {
        "protocol": "leave-one-out" if protocol == "loo" else f"{len(groups)}-fold",
        "protocol_id": protocol,
        "folds": int(len(groups)),
        "n": int(n),
        "thresholds": per_threshold,
        "mean_brier": mean_brier,
        "reference_brier": reference,
        "brier_skill_score": skill,
        "n_thresholds_scored": len(scored),
        "n_thresholds_exact": len(exact_rows),
        "n_dropped": int(dropped),
        # Pooled over the thresholds that were scored, in threshold order, for
        # :func:`calibration_table`.  Empty when no threshold was scored, so a
        # caller can pass them straight through without a length check.
        "held_out_predicted": (np.concatenate(held_out_predicted)
                               if held_out_predicted
                               else np.empty(0, dtype=float)),
        "held_out_observed": (np.concatenate(held_out_observed)
                              if held_out_observed
                              else np.empty(0, dtype=float)),
    }


def calibration_table(predicted, observed, n_bins=5):
    """Predicted probability against observed frequency, in equal-count bins.

    The number a Brier score hides: a set of probabilities can score well
    overall while being systematically too confident in the middle of its
    range.  Binned by predicted probability, the observed frequency should
    follow the diagonal, and the gap is the miscalibration a user would feel as
    "the map said 30 per cent and it happened half the time".

    Returns ``(rows, max_gap)``.  Rows are ``(low, high, n, mean_predicted,
    observed_frequency)``, sorted by predicted probability.
    """
    p = np.asarray(predicted, dtype=float)
    o = np.asarray(observed, dtype=float)
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = p[ok], o[ok]
    if len(p) == 0:
        return [], 0.0

    order = np.argsort(p)
    chunks = np.array_split(order, max(1, min(int(n_bins), len(order))))
    rows = []
    gaps = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        low = float(p[chunk].min())
        high = float(p[chunk].max())
        mean_predicted = float(p[chunk].mean())
        frequency = float(o[chunk].mean())
        rows.append((low, high, int(len(chunk)), mean_predicted, frequency))
        gaps.append(abs(mean_predicted - frequency))
    return rows, (max(gaps) if gaps else 0.0)


def threshold_separation(thresholds, values):
    """The smallest gap between adjacent thresholds, against the field's range.

    Returned so the tool can warn before the fact rather than after: thresholds
    a thousandth of the range apart are separated by a handful of observations,
    their two indicators are almost the same vector, and the variograms fitted
    to them will differ mostly by noise.
    """
    t = np.asarray(thresholds, dtype=float)
    z = np.asarray(values, dtype=float)
    z = z[np.isfinite(z)]
    if len(t) < 2 or len(z) == 0:
        return float("inf"), float("inf")
    gaps = np.diff(np.sort(t))
    spread = float(z.max() - z.min())
    smallest = float(gaps.min())
    relative = smallest / spread if spread > 0 else float("inf")
    return smallest, relative


def max_lag_for(coords, requested=None):
    """Resolve the maximum lag distance the fit will use, once per run.

    Every threshold must be binned over the *same* distance axis, or the
    families chosen for two thresholds are being compared on different axes and
    the automatic selection is not a comparison at all.
    """
    if requested is not None and float(requested) > 0.0:
        return float(requested)
    return default_max_lag(coords)


def summarise_fit(fit):
    """The fields of a fitted indicator variogram the report and log quote."""
    return {
        "model": str(fit["model"]),
        "nugget": float(fit["nugget"]),
        "psill": float(fit["psill"]),
        "sill": float(fit["sill"]),
        "range": float(fit["range"]),
        "r2": float(fit["r2"]) if fit.get("r2") is not None else float("nan"),
        "rmse": float(fit["rmse"]),
        "bins_used": int(fit["bins_used"]),
    }


def median_parameter(fits, key):
    """The median of one fitted parameter across thresholds.

    A median rather than a mean because a threshold near a tail can produce a
    degenerate fit - a range pinned to the last bin - and one such threshold
    should not drag the summary of the other four.
    """
    values = [float(f[key]) for f in fits if f and np.isfinite(f.get(key, math.nan))]
    if not values:
        return float("nan")
    return float(np.median(values))
