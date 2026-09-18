# -*- coding: utf-8 -*-
"""Empirical Bayesian Kriging: an automatic variogram, with its uncertainty.

Ordinary kriging takes the fitted semivariogram as true.  It is not: it is an
estimate from a finite sample, and two datasets from the same field would give
two different nuggets, sills and ranges.  The kriging variance that comes out
of the ordinary system describes the placement of the observation locations,
not the uncertainty in the model those observations were used to fit.  Where
the range is weakly identified - which, for a finite sample, is most of the
time - that omission understates the real prediction error, and it does so
without any visible symptom: the surface and its variance both look fine.

Empirical Bayesian Kriging (Krivoruchko 2012; Gribov & Krivoruchko 2020) closes
that gap by treating the semivariogram parameters as unknown and averaging over
them.  For each local subset of the observations:

1. Fit the semivariogram family to the subset's data - the best fit available
   before any uncertainty is considered.
2. Simulate a new dataset at those same locations from that fitted model, then
   fit the same family to the simulated data.  Because the simulation is a draw
   from the fitted model, the refitted parameters are a draw from the
   *distribution* of parameters the data could have produced, not a point.
3. Repeat ``n_sims`` times, each draw getting a Bayesian weight: the Gaussian
   likelihood of the real data under that draw's parameters.  A draw that
   describes the data well is weighted up; one that does not is weighted down.
4. Predict with every weighted draw and combine them.  The prediction is the
   weighted mean; the variance is the law of total variance over the mixture,
   so it carries both the kriging variance of each model and the spread
   *between* models.  That second term is the part ordinary kriging omits.

The consequence a reader should expect: EBK's variance surface is wider than
ordinary kriging's, and its cross-validation standardised errors should sit
closer to 1 than ordinary kriging's on the same data.  Both are claims that can
be checked rather than believed, and ``scratch/verify_ebk.py`` checks them.

What this module deliberately does NOT do, because a silent approximation here
would misreport the method rather than merely be imprecise:

* **The linear model is refused.**  A linear semivariogram is unbounded, so no
  stationary covariance exists for it and the Gaussian simulation in step 2 is
  undefined.  Simulating from a truncated linear model would be simulating from
  a different model than the one being reported, so the refusal is the honest
  option; supporting it properly needs the intrinsic random function machinery
  (ordinary kriging of generalised increments), which this module does not have.
* **The Bayes weight is the exact Gaussian likelihood**, not a proxy such as
  the weighted fit residual.  It costs one Cholesky factorisation per draw per
  subset, which is affordable at subset sizes; the fit residual would be
  cheaper but is not a probability and would make the weighted average
  something other than a posterior mean.
* **The draws are independent, not a Markov chain.**  Some descriptions of EBK
  resample from the previous draw's refit.  Independent draws from the first fit
  are what the ArcGIS documentation describes, are reproducible from a seed, and
  have no acceptance step whose correctness would then also have to be
  established.  The sampling distribution is therefore a proposal and the
  likelihood is what reweights it - importance sampling, stated as such.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .geostat_engines import (
    default_max_lag,
    detrend,
    fit_variogram,
    gamma,
    krige,
)

logger = logging.getLogger("PlanX GeoStats Lab")

#: Semivariogram families EBK can simulate from.  ``linear`` is absent on
#: purpose - see the module docstring; the tool's model enum is built from this
#: tuple so the two cannot drift apart.
EBK_MODELS = ("spherical", "exponential", "gaussian", "stable")

#: Draws per subset.  ArcGIS Pro's default is 100; below about 20 the weighted
#: mean is visibly noisy, and the cost is linear in this number.
EBK_DEFAULT_SIMULATIONS = 100
EBK_MAX_SIMULATIONS = 500

#: Observations per local subset, matching ArcGIS Pro's default.  0 means one
#: subset holding everything, which is the cleaner estimator but costs a kriging
#: solve per observation per draw; subsets are what make the method scale.
EBK_DEFAULT_SUBSET = 100

#: A subset is its own block plus the blocks around it, capped at this multiple
#: of the requested subset size, taking the points nearest the block centre.
SUBSET_CAP_FACTOR = 2.0

#: Below this many observations there is nothing to fit a variogram to.
EBK_MIN_POINTS = 8

#: Above this many observations the run is reported as slow rather than refused.
#: The cost is dominated by the per-draw solves, not by the fitting.
EBK_SOFT_LIMIT = 4000

#: Cross-validation refits and re-simulates the whole model once per fold, so
#: the affordable observation count is far lower than for the deterministic
#: interpolators, where a held-out point costs one kriging solve.  The value is
#: derived from the same cost model the run itself is bounded by rather than
#: guessed: folds x observations x simulations x local observations must stay
#: under the prediction's soft limit, which at the default five folds, 100
#: simulations and a 100-observation subset is satisfied up to about 480
#: observations.  Rounded to 500, and measured: a five-fold run over 387
#: observations at 30 simulations takes 3.4 s and at the 100-simulation default
#: takes 8.4 s, so the earlier limit of 300 refused validation for layers whose
#: validation costs seconds.
EBK_CV_LIMIT = 500
EBK_CV_FOLDS = 5

#: If fewer than this many draws can be refitted, the ensemble is abandoned and
#: the single initial fit is used, flagged so the report can say so.
EBK_MIN_FITS = 3

#: Diagonal jitter ladder for the Cholesky factorisation of a covariance matrix
#: built from near-coincident observation locations.  Starts relative to the
#: matrix's mean variance and grows by a factor of ten per attempt.
JITTER_START = 1e-10
JITTER_STEPS = 12

#: EBK's cost is the product of the number of target cells, the number of
#: simulations and the number of observations each simulation solves over, and
#: nothing else about it is expensive.  A timing sweep on a 3.14 GHz desktop puts
#: that product between 1.2e6 and 5.5e6 units per second: a single global subset
#: with 200 observations reaches the top of that range, while splitting the same
#: observations into many small subsets pays per-subset overhead and sits at the
#: bottom.  The rate below is the slow end deliberately - a guard that
#: under-estimates is worse than no guard - so the two limits are roughly twenty
#: seconds and three minutes of model time.  Unlike the deterministic
#: interpolators there is no optional stage to skip past a soft limit, so the
#: hard limit refuses and names the three knobs rather than appearing to hang.
EBK_WORK_PER_SECOND = 1.2e6
EBK_WORK_SOFT_LIMIT = 2.4e7
EBK_WORK_HARD_LIMIT = 2.4e8


def estimate_work(n_targets, n_sims, local_points):
    """The cost model behind the two limits above, in the same units."""
    return float(n_targets) * float(n_sims) * float(local_points)


def check_workload(n_targets, n_sims, local_points, where="Empirical Bayesian Kriging"):
    """Return a warning string when a run is slow, raise when it is hopeless.

    The refusal names all three factors with their current values, because all
    three are things the caller can change and "too slow" is not.
    """
    work = estimate_work(n_targets, n_sims, local_points)
    seconds = work / EBK_WORK_PER_SECOND
    if work > EBK_WORK_HARD_LIMIT:
        raise ValueError(
            f"{where} would need about {seconds:.0f} s for {int(n_targets):,} "
            f"target cells, {int(n_sims)} simulations per subset and "
            f"{int(local_points)} observations per simulation. Reduce the number "
            f"of cells by increasing the cell size, lower the number of "
            f"simulations, or lower the subset size: the cost is the product of "
            f"the three."
        )
    if work > EBK_WORK_SOFT_LIMIT:
        return (f"{int(n_targets):,} cells x {int(n_sims)} simulations x "
                f"{int(local_points)} observations is roughly {seconds:.0f} s "
                f"of computation.")
    return ""


# ── distance, binning and covariance ─────────────────────────────────────────

def _pair_distances(a, b=None):
    """Euclidean distances between two (n, 2) coordinate arrays."""
    other = a if b is None else np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    return np.hypot(a[:, None, 0] - other[None, :, 0],
                    a[:, None, 1] - other[None, :, 1])


def _bin_membership(d, n_bins, max_lag):
    """Which distance bin each pair falls in, plus the in-range mask.

    This reproduces the binning of ``geostat_engines.bin_pairs`` - equal-width
    bins from 0 to ``max_lag``, the first bin closed at the left - and exists
    separately only because the simulation loop re-bins the same distances a
    thousand times: computing the bin index once and reusing it turns that into
    a ``bincount``.  ``scratch/verify_ebk.py`` asserts the two agree exactly on
    random data, so the duplication is covered by a test rather than by hope.
    """
    edges = np.linspace(0.0, float(max_lag), int(n_bins) + 1)
    idx = np.digitize(d, edges) - 1
    keep = (idx >= 0) & (idx < int(n_bins))
    return idx, keep


def _binned_semivariance(sv, idx, keep, n_bins, lag_of_bin):
    """Mean semivariance per populated bin, given pre-computed bin membership.

    Returns the same ``(lag, semivariance, npairs)`` triple the engine's
    binning produces, with empty bins dropped by the same rule.
    """
    inside = idx[keep]
    counts = np.bincount(inside, minlength=int(n_bins))
    totals = np.bincount(inside, weights=sv[keep], minlength=int(n_bins))
    good = counts > 0
    return lag_of_bin[good], totals[good] / counts[good], counts[good]


def covariance_matrix(coords, model, nugget, psill, range_, alpha=1.5):
    """The stationary covariance implied by a semivariogram model.

    For this engine's convention - ``gamma(h) = nugget + psill * g(h)`` and
    ``gamma(0) = 0`` - the covariance is ``C(h) = sill - gamma(h)``, which is
    ``psill * (1 - g(h))`` away from the origin.  The diagonal is the total
    sill, which is what ``C(0)`` means for a process whose nugget is a
    discontinuity at the origin rather than a value there.  Setting it
    explicitly rather than relying on ``gamma(0) = 0`` keeps that convention in
    the code instead of in a footnote.

    The range parameter is spelled ``range_`` because ``rng`` in this module
    means a random generator, and a reader meeting ``rng=300.0`` here would
    reasonably read it as the wrong one of the two.
    """
    d = _pair_distances(coords)
    cov = float(nugget) + float(psill) - gamma(d, model, nugget, psill, range_,
                                               alpha)
    np.fill_diagonal(cov, float(nugget) + float(psill))
    return cov


def cholesky_with_jitter(cov):
    """Cholesky factor of a covariance matrix, adding diagonal jitter if needed.

    A covariance matrix built from observation locations is positive definite
    in exact arithmetic and can fail to be in floating point when two
    observations are a hair apart, or when the fitted range makes the matrix
    nearly singular.  Jitter is added to the diagonal in a growing ladder rather
    than assumed unnecessary, and the amount that was needed is returned so the
    report can show it: a large jitter is a statement that the fitted model
    barely holds together, which is worth seeing rather than hiding.

    Returns ``(factor, jitter_used)``.
    """
    cov = np.asarray(cov, dtype=float)
    n = len(cov)
    scale = float(np.trace(cov)) / max(n, 1)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    jitter = 0.0
    for attempt in range(JITTER_STEPS + 1):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(n)), jitter
        except np.linalg.LinAlgError:
            jitter = JITTER_START * scale if attempt == 0 else jitter * 10.0
    raise ValueError(
        "The covariance matrix implied by the fitted semivariogram is not "
        "positive definite even with a large diagonal jitter, which happens "
        "when observations coincide or when the fitted model is degenerate. "
        "Check the input layer for duplicate locations."
    )


# ── local subsets ────────────────────────────────────────────────────────────

def _quantile_edges(values, k):
    """``k + 1`` edges cutting ``values`` into k groups of roughly equal count."""
    v = np.asarray(values, dtype=float)
    if k <= 1:
        return np.array([float(v.min()), float(v.max())])
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, int(k) + 1)))
    if len(edges) < 2:
        span = float(v.max()) - float(v.min())
        edges = np.array([float(v.min()), float(v.min()) + max(span, 1e-9)])
    return edges


def spatial_subsets(coords, subset_size, overlap=1):
    """Partition observations into overlapping local subsets.

    ArcGIS Pro's EBK divides the data into overlapping subsets and fits a
    separate Bayesian model in each, which is what makes the method scale to
    observation counts no single kriging system could carry.  The partition here
    is equal-count rather than equal-area: blocks are cut at quantiles of the
    x and y coordinates, so a densely sampled quarter does not get a subset
    four times the size of a sparse one.

    A subset is its own block plus ``overlap`` rings of blocks around it, capped
    at ``SUBSET_CAP_FACTOR * subset_size`` points taken nearest the block's
    centre.  The overlap is what keeps a prediction that sits on a block
    boundary from being made by a model that has never seen the observations
    just across it.

    ``subset_size <= 0``, or an observation count at or below it, gives a single
    subset holding everything, which is the global model.

    Returns a dict: ``subsets`` (index arrays, in row-major block order),
    ``sizes``, ``cell_to_subset`` (a 2-D array over the partition grid),
    ``x_edges``, ``y_edges`` and ``gcols``/``grows``.  Every block, empty or
    not, maps to the subset of the nearest non-empty block, so any target -
    including one outside the observations' extent - resolves to a subset.
    """
    c = np.asarray(coords, dtype=float)
    n = len(c)
    size = int(subset_size)
    if n == 0:
        raise ValueError("There are no observations to partition.")

    if size <= 0 or n <= size:
        return {
            "subsets": [np.arange(n)],
            "sizes": [int(n)],
            "cell_to_subset": np.zeros((1, 1), dtype=int),
            "x_edges": np.array([float(c[:, 0].min()), float(c[:, 0].max())]),
            "y_edges": np.array([float(c[:, 1].min()), float(c[:, 1].max())]),
            "gcols": 1,
            "grows": 1,
            "overlap": 0,
        }

    wanted = int(math.ceil(n / float(size)))
    gcols = max(1, int(math.ceil(math.sqrt(wanted))))
    grows = max(1, int(math.ceil(wanted / float(gcols))))
    x_edges = _quantile_edges(c[:, 0], gcols)
    y_edges = _quantile_edges(c[:, 1], grows)
    gcols = len(x_edges) - 1
    grows = len(y_edges) - 1

    col_of = np.clip(np.digitize(c[:, 0], x_edges[1:-1]), 0, gcols - 1)
    row_of = np.clip(np.digitize(c[:, 1], y_edges[1:-1]), 0, grows - 1)

    cap = max(size, int(round(SUBSET_CAP_FACTOR * size)))
    ring = int(max(0, overlap))
    subsets = []
    cores = []
    for r in range(grows):
        for cc in range(gcols):
            here = (row_of == r) & (col_of == cc)
            if not here.any():
                continue
            if ring == 0:
                block = here
            else:
                block = np.zeros(n, dtype=bool)
                for dr in range(-ring, ring + 1):
                    rr = r + dr
                    if not (0 <= rr < grows):
                        continue
                    for dc in range(-ring, ring + 1):
                        ccc = cc + dc
                        if 0 <= ccc < gcols:
                            block |= (row_of == rr) & (col_of == ccc)
            idx = np.flatnonzero(block)
            if len(idx) > cap:
                # Keep the points nearest the block's centre, so the cap trims
                # the far side of the ring rather than an arbitrary corner.
                centre = np.array([c[here, 0].mean(), c[here, 1].mean()])
                order = np.argsort(
                    np.hypot(c[idx, 0] - centre[0], c[idx, 1] - centre[1]),
                    kind="stable",
                )
                idx = np.sort(idx[order[:cap]])
            cores.append((r, cc))
            subsets.append(idx)

    cell_to_subset = np.full((grows, gcols), -1, dtype=int)
    for k, (r, cc) in enumerate(cores):
        cell_to_subset[r, cc] = k
    # Empty blocks inherit the nearest fitted subset, measured between block
    # centres, so a prediction outside the sampled area is still made by the
    # model fitted closest to it rather than by an arbitrary one.
    core_rows = np.array([r for r, _ in cores], dtype=float)
    core_cols = np.array([cc for _, cc in cores], dtype=float)
    for r in range(grows):
        for cc in range(gcols):
            if cell_to_subset[r, cc] >= 0:
                continue
            gap = np.hypot(core_rows - r, core_cols - cc)
            cell_to_subset[r, cc] = int(np.argmin(gap))

    return {
        "subsets": subsets,
        "sizes": [int(len(s)) for s in subsets],
        "cell_to_subset": cell_to_subset,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "gcols": int(gcols),
        "grows": int(grows),
        "overlap": ring,
    }


def assign_subsets(targets, info):
    """The subset each target's prediction should come from."""
    t = np.atleast_2d(np.asarray(targets, dtype=float))
    if len(info["subsets"]) == 1:
        return np.zeros(len(t), dtype=int)
    col = np.clip(np.digitize(t[:, 0], info["x_edges"][1:-1]), 0, info["gcols"] - 1)
    row = np.clip(np.digitize(t[:, 1], info["y_edges"][1:-1]), 0, info["grows"] - 1)
    return info["cell_to_subset"][row, col]


# ── the ensemble ─────────────────────────────────────────────────────────────

def check_model(model):
    """Refuse a family this engine cannot simulate from, naming the reason.

    The linear model reaching the covariance builder would fail later, in the
    Cholesky factorisation, with a message about duplicate locations - which is
    the wrong explanation of the wrong problem.  Catching it at the boundary
    keeps the error actionable.
    """
    if model == "auto" or model in EBK_MODELS:
        return
    if model == "linear":
        raise ValueError(
            "The linear semivariogram cannot be used in Empirical Bayesian "
            "Kriging. It is unbounded, so no stationary covariance exists for "
            "it and the Gaussian simulation the method is built on is "
            "undefined. Simulating from a truncated linear model would be "
            "simulating from a different model than the one reported. Use "
            + ", ".join(EBK_MODELS) + "."
        )
    raise ValueError(
        f"Unknown semivariogram model for Empirical Bayesian Kriging: "
        f"'{model}'. Supported: {', '.join(EBK_MODELS)}, or 'auto'."
    )


def choose_family(lag, semivariance, npairs, alpha=1.5, candidates=EBK_MODELS):
    """The family whose Cressie-weighted fit cost is lowest.

    The weights depend only on the empirical semivariogram, so every family is
    judged on the same weighted residuals and the comparison is fair.  A caveat
    that the report repeats: the lowest cost is the best description of the
    binned semivariogram, which is not the same claim as the best predictor -
    the range and the nugget trade off against each other, so two families can
    fit almost equally well and predict differently.
    """
    best = None
    for family in candidates:
        try:
            fit = fit_variogram(lag, semivariance, npairs, family, alpha)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or fit["cost"] < best["cost"]:
            best = fit
    if best is None:
        raise ValueError(
            "None of the semivariogram families could be fitted to this "
            "subset's empirical variogram. The subset may be too small or too "
            "nearly collinear; increase the subset size or the bin count."
        )
    return best


def fit_ensemble(coords, values, model, alpha=1.5,
                 n_sims=EBK_DEFAULT_SIMULATIONS, seed=0, n_bins=15,
                 drift="none", progress=None):
    """One local Bayesian ensemble: fit, simulate, refit, reweight.

    ``model`` is a family name or ``"auto"``, in which case the family is
    chosen once from this subset's own empirical variogram and every draw then
    uses that family.  Choosing per draw would multiply the fitting cost by the
    number of families and is not what the method asks for: the family is the
    modelling decision, the parameters are what carry uncertainty.

    Returns a dict with the initial fit, the surviving draws, their normalised
    weights and the summary statistics the report shows.  ``fallback`` is True
    when too few draws could be refitted, in which case the initial fit carries
    all the weight and the caller must say so rather than present a single model
    as an ensemble.
    """
    c = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    n = len(z)
    if n < EBK_MIN_POINTS:
        raise ValueError(
            f"Empirical Bayesian Kriging needs at least {EBK_MIN_POINTS} "
            f"observations in a subset to fit a semivariogram; got {n}. Use a "
            "larger subset size."
        )
    n_sims = int(n_sims)
    if n_sims < 1:
        raise ValueError("At least one simulation is required.")
    if n_sims > EBK_MAX_SIMULATIONS:
        raise ValueError(
            f"{n_sims} simulations exceeds the cap of {EBK_MAX_SIMULATIONS}. "
            "The cost is linear in the simulation count and beyond a few "
            "hundred draws the weighted mean stops changing."
        )
    check_model(model)

    max_lag = default_max_lag(c)
    if not np.isfinite(max_lag) or max_lag <= 0.0:
        raise ValueError(
            "Every observation in this subset shares one location, so there "
            "are no distances to build a semivariogram from."
        )

    i, j = np.triu_indices(n, 1)
    d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
    sv = 0.5 * (z[i] - z[j]) ** 2
    idx, keep = _bin_membership(d, n_bins, max_lag)
    lag_all = np.zeros(int(n_bins))
    counts_all = np.bincount(idx[keep], minlength=int(n_bins))
    sums_all = np.bincount(idx[keep], weights=d[keep], minlength=int(n_bins))
    for b in np.flatnonzero(counts_all > 0):
        lag_all[b] = sums_all[b] / counts_all[b]
    lag, semivariance, npairs = _binned_semivariance(sv, idx, keep, n_bins, lag_all)
    if len(lag) < 4:
        raise ValueError(
            f"This subset has only {len(lag)} populated distance bins, and at "
            "least 4 are needed to fit a semivariogram. Increase the subset "
            "size or the number of bins, or reduce the number of subsets."
        )

    family = model
    if family == "auto":
        family = str(choose_family(lag, semivariance, npairs, alpha)["model"])
    initial = fit_variogram(lag, semivariance, npairs, family, alpha)

    residuals, _coefficients = detrend(c, z, drift)

    rng = np.random.default_rng(int(seed))
    cov0 = covariance_matrix(c, family, initial["nugget"], initial["psill"],
                             initial["range"], alpha)
    factor0, jitter0 = cholesky_with_jitter(cov0)
    jitter_max = jitter0

    members = []
    n_failed = 0
    for _ in range(n_sims):
        realisation = factor0 @ rng.standard_normal(n)
        sv_sim = 0.5 * (realisation[i] - realisation[j]) ** 2
        # The bin edges are the data's, so the refit sees the same distance
        # axis as the initial fit and the two are comparable.  A realisation
        # binned on its own edges would be fitted on a different scale.
        _, gam_sim, cnt_sim = _binned_semivariance(
            sv_sim, idx, keep, n_bins, lag_all)
        try:
            member = fit_variogram(lag, gam_sim, cnt_sim, family, alpha)
        except (ValueError, np.linalg.LinAlgError):
            # A realisation whose variogram is flat carries no structure to
            # fit. Dropping it is correct; dropping all of them is not, and
            # the fallback below is what keeps that visible.
            n_failed += 1
            continue
        cov_s = covariance_matrix(c, family, member["nugget"], member["psill"],
                                  member["range"], alpha)
        factor_s, jitter_s = cholesky_with_jitter(cov_s)
        jitter_max = max(jitter_max, jitter_s)
        y = np.linalg.solve(factor_s, residuals)
        logdet = 2.0 * float(np.sum(np.log(np.diag(factor_s))))
        member["loglik"] = -0.5 * (
            n * math.log(2.0 * math.pi) + logdet + float(y @ y))
        members.append(member)

    fallback = len(members) < EBK_MIN_FITS
    if fallback:
        members = [dict(initial)]
        for m in members:
            m["loglik"] = 0.0

    loglik = np.asarray([m["loglik"] for m in members], dtype=float)
    loglik = loglik - loglik.max()
    weights = np.exp(loglik)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        weights = np.ones(len(members), dtype=float) / len(members)
    else:
        weights = weights / total

    def _stats(column):
        """Weighted mean and standard deviation of one fitted parameter.

        Taken over the draws directly rather than derived from the other
        parameters' moments: the nugget and the partial sill trade off against
        each other, so combining their spreads as if they were independent
        would misstate the spread of the sill, which is the sum of the two.
        """
        arr = np.asarray(column, dtype=float)
        mean = float(np.sum(weights * arr))
        var = float(np.sum(weights * arr * arr)) - mean * mean
        return mean, math.sqrt(max(var, 0.0))

    nugget_s = _stats([m["nugget"] for m in members])
    psill_s = _stats([m["psill"] for m in members])
    range_s = _stats([m["range"] for m in members])
    sill_s = _stats([m["nugget"] + m["psill"] for m in members])

    return {
        "family": family,
        "requested_model": model,
        "n": int(n),
        "initial": initial,
        "members": members,
        "weights": weights,
        "n_sims": n_sims,
        "n_failed": int(n_failed),
        "fallback": bool(fallback),
        "max_lag": float(max_lag),
        "bins": int(len(lag)),
        "max_weight": float(weights.max()),
        "effective_models": float(1.0 / np.sum(weights ** 2)),
        "jitter": float(jitter_max),
        "posterior": {"nugget": nugget_s[0], "psill": psill_s[0],
                      "sill": sill_s[0], "range": range_s[0]},
        "spread": {"nugget": nugget_s[1], "psill": psill_s[1],
                   "sill": sill_s[1], "range": range_s[1]},
    }


def ensemble_summary(ensemble):
    """The ensemble's fitted parameters and their spread, for a report row."""
    return {
        "family": ensemble["family"],
        "n": ensemble["n"],
        "bins": ensemble["bins"],
        "max_lag": ensemble["max_lag"],
        "initial": ensemble["initial"],
        "posterior": dict(ensemble["posterior"]),
        "spread": dict(ensemble["spread"]),
        "members": len(ensemble["members"]),
        "n_failed": ensemble["n_failed"],
        "fallback": ensemble["fallback"],
        "effective_models": ensemble["effective_models"],
        "max_weight": ensemble["max_weight"],
        "jitter": ensemble["jitter"],
    }


# ── prediction ───────────────────────────────────────────────────────────────

def ebk_predict(coords, values, targets, model="auto", alpha=1.5,
                n_sims=EBK_DEFAULT_SIMULATIONS, subset_size=EBK_DEFAULT_SUBSET,
                overlap=1, seed=0, n_bins=15, drift="none", progress=None):
    """Predict at ``targets`` by averaging over fitted-and-simulated models.

    Every target is served by exactly one subset - the one whose block contains
    it - so the run costs one ensemble per subset rather than one per cell.  The
    return is ``(prediction, variance, diagnostics)``; the diagnostics carry the
    per-subset ensembles, which is what lets the report show which model was
    fitted where instead of asserting that an ensemble was used.

    The variance is the law of total variance over the weighted draws,
    ``sum_s w_s (var_s + pred_s^2) - pred^2``, so it is never smaller than the
    weighted mean of the individual kriging variances.  That difference is the
    whole point of the method: it is the part of the prediction uncertainty
    that comes from not knowing the variogram.

    At an observed location every draw is exact and reports zero variance, so
    the mixture does too - EBK is an exact interpolator, like the kriging it is
    built on.
    """
    c = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    t = np.atleast_2d(np.asarray(targets, dtype=float))
    check_model(model)
    if len(c) < EBK_MIN_POINTS:
        raise ValueError(
            f"Empirical Bayesian Kriging needs at least {EBK_MIN_POINTS} "
            f"observations; got {len(c)}."
        )
    if len(t) == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, {"n_subsets": 0, "subsets": [], "targets": 0}

    info = spatial_subsets(c, subset_size, overlap)
    owner = assign_subsets(t, info)

    mean_local = float(np.mean(info["sizes"])) if info["sizes"] else float(len(c))
    workload = check_workload(len(t), n_sims, mean_local)
    if workload:
        logger.warning("GeoStats Lab: %s", workload)

    prediction = np.zeros(len(t), dtype=float)
    variance = np.zeros(len(t), dtype=float)
    per_subset = []
    n_subsets = len(info["subsets"])
    for k, rows in enumerate(info["subsets"]):
        if progress is not None and progress(k, n_subsets) is False:
            raise RuntimeError("Empirical Bayesian Kriging cancelled.")
        target_mask = owner == k
        here = np.flatnonzero(target_mask)
        ensemble = fit_ensemble(c[rows], z[rows], model, alpha, n_sims,
                                seed=seed + k, n_bins=n_bins, drift=drift)
        summary = ensemble_summary(ensemble)
        summary["target_cells"] = int(len(here))
        if len(here):
            local_targets = t[here]
            acc_pred = np.zeros(len(here), dtype=float)
            acc_var = np.zeros(len(here), dtype=float)
            acc_sq = np.zeros(len(here), dtype=float)
            for member, weight in zip(ensemble["members"], ensemble["weights"]):
                p, v = krige(c[rows], z[rows], local_targets, member["model"],
                             member["nugget"], member["psill"], member["range"],
                             drift, member["alpha"])
                acc_pred += weight * p
                acc_var += weight * v
                acc_sq += weight * p * p
            prediction[here] = acc_pred
            variance[here] = np.maximum(acc_var + acc_sq - acc_pred * acc_pred, 0.0)
        per_subset.append(summary)

    if progress is not None and progress(n_subsets, n_subsets) is False:
        raise RuntimeError("Empirical Bayesian Kriging cancelled.")

    diagnostics = {
        "n_subsets": n_subsets,
        "subset_sizes": list(info["sizes"]),
        "targets": int(len(t)),
        "gamma": [int(info["gcols"]), int(info["grows"])],
        "overlap": int(info["overlap"]),
        "subsets": per_subset,
        "families": sorted({s["family"] for s in per_subset}),
        "n_failed": int(sum(s["n_failed"] for s in per_subset)),
        "fallbacks": int(sum(1 for s in per_subset if s["fallback"])),
        "effective_models": float(np.mean([s["effective_models"]
                                           for s in per_subset])),
        "min_effective_models": float(min(s["effective_models"]
                                          for s in per_subset)),
        "max_weight": float(max(s["max_weight"] for s in per_subset)),
        "jitter": float(max(s["jitter"] for s in per_subset)),
        "requested_model": model,
        "n_sims": int(n_sims),
        "seed": int(seed),
        "subset_size": int(subset_size),
        "work_estimate": float(estimate_work(len(t), n_sims, mean_local)),
        "work_warning": workload,
    }
    return prediction, variance, diagnostics


def ebk_fold_cross_validation(coords, values, model="auto", alpha=1.5,
                              n_sims=EBK_DEFAULT_SIMULATIONS,
                              subset_size=EBK_DEFAULT_SUBSET, overlap=1,
                              folds=EBK_CV_FOLDS, seed=0, n_bins=15,
                              drift="none", progress=None):
    """k-fold cross-validation of the whole EBK model.

    Leave-one-out is not what is done here, and the difference is not a detail.
    A held-out point costs EBK a complete refit - an ensemble of simulations and
    a Cholesky factorisation per draw - rather than the single kriging solve it
    costs ordinary kriging, so leave-one-out at any useful observation count is
    out of reach.  k-fold is affordable, but each fold's model sees only part of
    the data, so its errors are systematically larger than leave-one-out's and
    the two are not comparable.  The report states the protocol for this
    reason; a number with no protocol attached is not a validation result.
    """
    c = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    n = len(z)
    check_model(model)
    folds = int(folds)
    if folds < 2:
        raise ValueError("Cross-validation needs at least two folds.")
    if n < EBK_MIN_POINTS * 2:
        raise ValueError(
            f"Cross-validation needs at least {EBK_MIN_POINTS * 2} "
            f"observations so that every fold still has {EBK_MIN_POINTS} to "
            f"fit on; got {n}."
        )
    if folds > n - EBK_MIN_POINTS:
        folds = max(2, n - EBK_MIN_POINTS)

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    groups = [order[k::folds] for k in range(folds)]

    errors = np.zeros(n, dtype=float)
    standardised = np.zeros(n, dtype=float)
    for k, held in enumerate(groups):
        if progress is not None and progress(k, folds) is False:
            raise RuntimeError("Empirical Bayesian Kriging cancelled.")
        keep = np.ones(n, dtype=bool)
        keep[held] = False
        prediction, variance, _diag = ebk_predict(
            c[keep], z[keep], c[held], model, alpha, n_sims, subset_size,
            overlap, seed=seed + 1000 * (k + 1), n_bins=n_bins, drift=drift,
        )
        errors[held] = z[held] - prediction
        standardised[held] = errors[held] / np.sqrt(np.maximum(variance, 1e-300))

    std = float(np.std(z))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {
        "n": int(n),
        "n_scored": int(n),
        "n_dropped": 0,
        "folds": folds,
        "protocol": f"{folds}-fold",
        "protocol_id": "kfold",
        "me": float(errors.mean()),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(errors))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "mse": float(standardised.mean()),
        "rmsse": float(np.sqrt(np.mean(standardised ** 2))),
        "nrmse": (rmse / std) if std > 0 else None,
        "field_std": std,
        "errors": errors,
        "standardized": standardised,
    }
