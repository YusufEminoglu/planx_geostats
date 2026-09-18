# -*- coding: utf-8 -*-
"""Geostatistical engines: semivariogram modelling and kriging.

numpy-only pure functions, matching the convention of stats_engines.py -- no
QGIS import, so this module is importable and testable without a QGIS runtime.

Conventions this module commits to, because the alternatives are also in use
and a reader cannot tell them apart from the numbers alone:

* **Semivariogram.** ``gamma(h) = nugget + psill * g(h)`` for ``h > 0`` and
  ``gamma(0) = 0`` by definition.  The nugget is therefore a *discontinuity at
  the origin*, not a value at the origin.
* **Kriging system.** The matrix diagonal is 0, following ``gamma(0) = 0``.
  The consequence is the standard GSLIB / PyKrige / gstat ordinary-kriging
  behaviour: a prediction made AT a datum that is in the system returns that
  datum exactly and reports zero variance.  A nugget shows up where it
  matters -- predicting at a location that is *not* in the system -- and it
  leaves leave-one-out cross-validation correctly calibrated either way,
  because there the target is never a datum.  Both facts are pinned by tests
  in ``scratch/geostat_proto.py``; see also the shortcoming register.
* **Weighting.** Model fitting minimises the Cressie (1985) weighted sum of
  squares, ``w(h) = N(h) / gamma(h)^2``, which is scale-invariant and
  downweights the noisy long-lag bins.  The weights are normalised before use,
  so the reported ``cost`` is not a chi-square statistic.
* **Range.** Fitted as the *practical* range under the 3/a convention for the
  exponential, gaussian and stable models, i.e. the lag at which about 95 per
  cent of the sill is reached, which puts every model's range on the same
  footing as the spherical model's true range.
* **Variance at a distance.** Ordinary-kriging variance converges to
  ``sill + a non-negative mean-uncertainty term``, NOT to the sill alone --
  that limit belongs to *simple* kriging.  So a variance surface legitimately
  shows values ABOVE the fitted sill, growing towards the map edges.  This
  looks like a bug and is not one; verified against a simple-kriging oracle
  in ``scratch/verify_engine.py``, where SK reaches the sill exactly and OK
  sits above it.  The variance surface is therefore a relative statement
  about prediction error, not an absolute one.

A caveat that belongs in the open rather than in a footnote: the range is
*weakly identified* from a finite sample, because it trades off against the
nugget and the sill.  A fitted curve can track the empirical variogram
closely (R^2 > 0.9) while the range is still 20-30 per cent away from the
truth.  Fit quality and parameter identifiability are not the same thing, and
the report says so.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger("PlanX GeoStats Lab")

#: Supported theoretical semivariogram models.
MODELS = ("spherical", "exponential", "gaussian", "linear", "stable")

#: Supported drift orders for universal kriging ("none" = ordinary kriging).
DRIFTS = ("none", "linear", "quadratic")

#: Above this many input points the O(n^3) kriging solve stops being
#: interactive and the caller should warn.  A soft limit, not a cap.
KRIGING_SOFT_LIMIT = 2000

#: Above this many points leave-one-out cross-validation needs n kriging
#: solves and should be reported as slow rather than silently run.
LOO_SOFT_LIMIT = 600


# ── variogram models ─────────────────────────────────────────────────────────

def _unit_g(h, rng, model, alpha=1.5):
    """The correlation-shape term g(h) in [0, 1], with g(0) = 0."""
    u = np.asarray(h, dtype=float) / max(float(rng), 1e-12)
    if model == "spherical":
        return np.where(u < 1.0, 1.5 * u - 0.5 * u ** 3, 1.0)
    if model == "exponential":
        return 1.0 - np.exp(-3.0 * u)
    if model == "gaussian":
        return 1.0 - np.exp(-3.0 * u * u)
    if model == "linear":
        return u
    if model == "stable":
        return 1.0 - np.exp(-3.0 * np.power(u, float(alpha)))
    raise ValueError(f"Unknown semivariogram model: {model}")


def gamma(h, model, nugget, psill, rng, alpha=1.5):
    """Semivariogram value(s).  ``gamma(0) = 0``; the nugget applies to h > 0."""
    h = np.asarray(h, dtype=float)
    return np.where(h <= 0.0, 0.0,
                    float(nugget) + float(psill) * _unit_g(h, rng, model, alpha))


def _drift_basis(coords, drift):
    """Polynomial drift basis.  p = 1, 3 or 6 columns."""
    c = np.asarray(coords, dtype=float)
    if drift == "linear":
        return np.column_stack([np.ones(len(c)), c[:, 0], c[:, 1]])
    if drift == "quadratic":
        return np.column_stack([np.ones(len(c)), c[:, 0], c[:, 1],
                                c[:, 0] ** 2, c[:, 1] ** 2, c[:, 0] * c[:, 1]])
    raise ValueError(f"Unknown drift: {drift}")


def detrend(coords, values, drift):
    """Remove a fitted polynomial drift from ``values`` by least squares.

    Universal kriging fits the variogram to the *residuals* left after the
    drift, not to the raw values.  Skipping this step is not a small
    approximation: a trend inflates the long-lag bins, so the fitted model is
    still rising at the last bin and settles on a range that describes the trend
    rather than the field.  Measured on a test field whose true range was 220 map
    units, the linear trend used in the plugin's own runtime test pushed the
    fitted range to 513 and made universal kriging predict *worse* than ordinary
    kriging; fitting on residuals recovered 238, and universal kriging then
    predicted better, as it should.

    Returns ``(residuals, coefficients)``.  With ``drift="none"`` the drift is
    the mean alone, so the residuals are the mean-centred values.
    """
    z = np.asarray(values, dtype=float)
    basis = (np.ones((len(z), 1)) if drift == "none"
             else _drift_basis(coords, drift))
    coefficients, *_ = np.linalg.lstsq(basis, z, rcond=None)
    return z - basis @ coefficients, coefficients


def _pair_dist(a, b):
    """Euclidean distance matrix between two (n, 2) coordinate arrays."""
    return np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])


def default_max_lag(coords):
    """Half the maximum pairwise separation.

    Journel & Huijbregts (1978) give half the maximum separation as the
    conventional upper bound: beyond it the number of contributing pairs
    collapses and the empirical variogram becomes noise.  Returns 0.0 when
    every point coincides, which callers must treat as "cannot bin".
    """
    c = np.asarray(coords, dtype=float)
    if len(c) < 2:
        return 0.0
    i, j = np.triu_indices(len(c), 1)
    return 0.5 * float(np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1]).max())


# ── empirical semivariogram ──────────────────────────────────────────────────

def _pair_cloud(coords, values):
    """Upper-triangle pairwise distances and semivariances."""
    c = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    i, j = np.triu_indices(len(z), 1)
    d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
    sv = 0.5 * (z[i] - z[j]) ** 2
    return d, sv


def _bin_pairs(d, sv, n_bins=15, max_lag=None):
    """Bin already-computed pair distances and semivariances.

    Kept separate from the pairing so a direction filter can select *pairs*
    before binning.  Filtering the points instead would silently re-pair the
    survivors among themselves, which is a different and wrong variogram.
    """
    empty = np.zeros(0)
    if len(d) == 0:
        return empty, empty, np.zeros(0, dtype=int)
    if max_lag is None:
        max_lag = float(np.max(d)) / 2.0
    max_lag = float(max_lag)
    if not np.isfinite(max_lag) or max_lag <= 0.0:
        return empty, empty, np.zeros(0, dtype=int)

    n_bins = int(n_bins)
    edges = np.linspace(0.0, max_lag, n_bins + 1)
    idx = np.digitize(d, edges) - 1
    keep = (idx >= 0) & (idx < n_bins)
    lag = np.zeros(n_bins)
    gam = np.zeros(n_bins)
    cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = keep & (idx == b)
        cnt[b] = int(m.sum())
        if cnt[b]:
            lag[b] = d[m].mean()
            gam[b] = sv[m].mean()
    good = cnt > 0
    return lag[good], gam[good], cnt[good]


def empirical_variogram(coords, values, n_bins=15, max_lag=None):
    """Binned empirical semivariogram.

    Returns ``(lag, semivariance, npairs, distances, semivariances)``: the
    three populated-bin arrays, plus the raw pair cloud so a caller can draw
    it without recomputing the O(n^2) pairing.  Bins with no pairs are
    dropped, so the first three arrays can be shorter than ``n_bins``.
    """
    d, sv = _pair_cloud(coords, values)
    lag, gam, cnt = _bin_pairs(d, sv, n_bins, max_lag)
    return lag, gam, cnt, d, sv


def _cone_mask(coords, azimuth_deg, tolerance_deg):
    """Boolean mask over the upper-triangle pairs inside a direction cone.

    Directions use the mathematical convention: 0 deg points along +x (east),
    angles increase counter-clockwise, and bearings are folded modulo 180 so
    a pair and its reverse land in the same cone and the variogram stays
    symmetric under swapping the pair.
    """
    c = np.asarray(coords, dtype=float)
    _, j = np.triu_indices(len(c), 1)
    i = np.triu_indices(len(c), 1)[0]
    dx = c[j, 0] - c[i, 0]
    dy = c[j, 1] - c[i, 1]
    bearing = np.degrees(np.arctan2(dy, dx)) % 180.0
    target = float(azimuth_deg) % 180.0
    delta = np.abs(bearing - target)
    delta = np.minimum(delta, 180.0 - delta)
    return delta <= float(tolerance_deg)


def directional_variogram(coords, values, azimuth_deg, tolerance_deg=22.5,
                          n_bins=12, max_lag=None):
    """Empirical semivariogram built only from pairs inside a direction cone.

    Returns ``(lag, semivariance, npairs)``.  Pass the same ``max_lag`` for
    every direction when comparing them; a per-direction default would give
    each cone its own distance axis and the comparison would be meaningless.
    """
    d, sv = _pair_cloud(coords, values)
    mask = _cone_mask(coords, azimuth_deg, tolerance_deg)
    return _bin_pairs(d[mask], sv[mask], n_bins, max_lag)


def anisotropy_rose(coords, values, n_directions=8, tolerance_deg=22.5,
                    n_bins=12, max_lag=None):
    """Directional range estimates, for the variogram rose.

    For each direction the directional variogram is fitted and the fitted
    range is reported.  ``max_lag`` is resolved once from the full pair cloud
    and reused for every direction, so all the cones share one distance axis.

    Where a direction has too few pairs to fit, the entry is NaN and the
    caller must say so rather than draw a zero.  A rose whose ranges vary
    systematically with direction is evidence of anisotropy, which this
    plugin's kriging does NOT model -- see the shortcoming register.
    """
    d, _ = _pair_cloud(coords, values)
    if len(d) == 0:
        angles = np.linspace(0.0, 180.0, int(n_directions), endpoint=False)
        return {"angles_deg": angles,
                "ranges": np.full(int(n_directions), np.nan),
                "max_min_ratio": float("nan")}
    if max_lag is None:
        max_lag = float(d.max()) / 2.0

    angles = np.linspace(0.0, 180.0, int(n_directions), endpoint=False)
    ranges = np.full(int(n_directions), np.nan)
    for k, ang in enumerate(angles):
        lag, gam, cnt = directional_variogram(
            coords, values, ang, tolerance_deg, n_bins, max_lag)
        if len(lag) < 4:
            continue
        try:
            fit = fit_variogram(lag, gam, cnt, "spherical")
        except (ValueError, np.linalg.LinAlgError):
            continue
        ranges[k] = fit["range"]
    finite = ranges[np.isfinite(ranges)]
    ratio = (float(finite.max() / finite.min())
             if len(finite) >= 2 and finite.min() > 0 else float("nan"))
    return {"angles_deg": angles, "ranges": ranges, "max_min_ratio": ratio}


# ── model fitting (Levenberg-Marquardt, numpy only) ──────────────────────────

def fit_variogram(lag, semivariance, npairs, model="spherical", alpha=1.5,
                  iters=300, weights="cressie"):
    """Fit nugget, partial sill and range by weighted least squares.

    Returns a dict with ``model``, ``nugget``, ``psill``, ``sill``, ``range``,
    ``alpha``, ``r2``, ``rmse``, ``cost`` and ``bins_used``.  The optimisation
    runs in log-parameter space, so the fitted nugget, sill and range are
    positive by construction rather than by clamping.
    """
    lag = np.asarray(lag, dtype=float)
    semivariance = np.asarray(semivariance, dtype=float)
    npairs = np.asarray(npairs, dtype=float)
    ok = (semivariance > 0) & (npairs > 0) & np.isfinite(semivariance)
    lag, semivariance, npairs = lag[ok], semivariance[ok], npairs[ok]
    if len(lag) < 4:
        raise ValueError(
            "At least 4 populated distance bins are needed to fit a "
            f"semivariogram model; got {len(lag)}. Increase the lag "
            "tolerance, the bin count, or the maximum lag distance."
        )

    if weights == "cressie":
        w = npairs / np.maximum(semivariance, 1e-12) ** 2
    else:
        w = npairs.copy()
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        w = np.ones_like(w)
        total = float(w.sum())
    w = w / total

    sill0 = float(semivariance.max())
    if not np.isfinite(sill0) or sill0 <= 0.0:
        raise ValueError("The empirical semivariogram is flat at zero; "
                         "there is no structure to fit.")
    reached = np.flatnonzero(semivariance >= 0.9 * sill0)
    rng0 = float(lag[reached[0]]) if len(reached) else float(lag[-1])
    theta = np.log([max(0.1 * sill0, 1e-9), max(0.9 * sill0, 1e-9),
                    max(rng0, 1e-9)])

    def residual(th):
        n0, ps, rg = np.exp(th)
        return (gamma(lag, model, n0, ps, rg, alpha) - semivariance) * np.sqrt(w)

    r = residual(theta)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(int(iters)):
        jac = np.zeros((len(lag), 3))
        for k in range(3):
            probe = theta.copy()
            probe[k] += 1e-7
            jac[:, k] = (residual(probe) - r) / 1e-7
        try:
            step = np.linalg.solve(jac.T @ jac + lam * np.eye(3), -(jac.T @ r))
        except np.linalg.LinAlgError:
            break
        trial = theta + step
        r_trial = residual(trial)
        cost_trial = float(r_trial @ r_trial)
        if np.isfinite(cost_trial) and cost_trial < cost:
            theta, r, cost = trial, r_trial, cost_trial
            lam = max(lam * 0.3, 1e-10)
        else:
            lam *= 4.0
            if lam > 1e10:
                break

    n0, ps, rg = np.exp(theta)
    fitted = gamma(lag, model, n0, ps, rg, alpha)
    ss_res = float(np.sum(w * (semivariance - fitted) ** 2))
    mean_w = float(np.average(semivariance, weights=w))
    ss_tot = float(np.sum(w * (semivariance - mean_w) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "model": model,
        "nugget": float(n0),
        "psill": float(ps),
        "sill": float(n0 + ps),
        "range": float(rg),
        "alpha": float(alpha),
        "r2": r2,
        "rmse": math.sqrt(float(np.mean((semivariance - fitted) ** 2))),
        "cost": cost,
        "bins_used": int(len(lag)),
    }


def fitted_curve(model, nugget, psill, rng, h_max, n=120, alpha=1.5):
    """Return ``(h, gamma)`` for drawing the fitted model over a plot."""
    h = np.linspace(0.0, max(float(h_max), 0.0), int(n))
    return h, gamma(h, model, nugget, psill, rng, alpha)


#: Attribute names of the fitted-parameter layer the semivariogram tool writes
#: and the kriging tool reads.  Both sides import this tuple rather than each
#: repeating a list, so the handoff contract is defined once: a mismatch would
#: otherwise surface only as a silently wrong kriging model, not as an error.
PARAM_LAYER_FIELDS = (
    "model", "nugget", "psill", "sill", "range", "alpha",
    "r2", "rmse", "bins", "npoints", "dir_ratio",
)


def model_to_raster_contract(params):
    """The parameter contract shared with the kriging tool.

    Both tools carry the same five keys, which is what makes "run the
    semivariogram tool, then krige with what it found" a real handoff instead
    of a copy-by-eye exercise.  Kept as one function so the two sides cannot
    drift apart.
    """
    return {
        "model": str(params["model"]),
        "nugget": float(params["nugget"]),
        "psill": float(params["psill"]),
        "range": float(params["range"]),
        "alpha": float(params.get("alpha", 1.5)),
    }


# ── kriging ──────────────────────────────────────────────────────────────────

def krige(coords, values, targets, model, nugget, psill, rng, drift="none",
          alpha=1.5):
    """Ordinary (``drift="none"``) or universal kriging, batched over targets.

    The kriging matrix is assembled and factorised once and every prediction
    point is a column of the right-hand side, so the whole surface costs a
    single ``numpy.linalg.solve`` rather than one solve per cell.

    Returns ``(prediction, variance)`` as arrays aligned with ``targets``.
    """
    c = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    t = np.atleast_2d(np.asarray(targets, dtype=float))
    n = len(z)
    m = len(t)
    if n < 2:
        raise ValueError("Kriging needs at least 2 observations.")
    if drift not in DRIFTS:
        raise ValueError(f"Unknown drift: {drift}")

    g = gamma(_pair_dist(c, c), model, nugget, psill, rng, alpha)
    np.fill_diagonal(g, 0.0)

    if drift == "none":
        a = np.zeros((n + 1, n + 1))
        a[:n, :n] = g
        a[:n, n] = 1.0
        a[n, :n] = 1.0
    else:
        f = _drift_basis(c, drift)
        p = f.shape[1]
        a = np.zeros((n + p, n + p))
        a[:n, :n] = g
        a[:n, n:] = f
        a[n:, :n] = f.T

    g0 = gamma(_pair_dist(c, t), model, nugget, psill, rng, alpha)

    if drift == "none":
        b = np.vstack([g0, np.ones((1, m))])
    else:
        b = np.vstack([g0, _drift_basis(t, drift).T])

    try:
        x = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The kriging system is singular. This usually means duplicate "
            "points at identical coordinates carrying different values, or a "
            "drift order too high for the number of observations."
        ) from exc

    w = x[:n]
    pred = w.T @ z
    if drift == "none":
        var = np.einsum("ij,ij->j", w, g0) + x[n]
    else:
        var = (np.einsum("ij,ij->j", w, g0)
               + np.einsum("ik,ik->k", x[n:], _drift_basis(t, drift).T))
    return pred, np.maximum(var, 0.0)


def krige_chunked(coords, values, targets, model, nugget, psill, rng,
                  drift="none", alpha=1.5, max_rhs_bytes=64 * 1024 * 1024,
                  progress=None):
    """``krige`` over a target grid in chunks, bounding peak memory.

    ``krige`` solves for every target at once, which is the fast way to predict
    a few thousand cells but cannot scale to a raster: the right-hand side alone
    is ``(n + p) * len(targets)`` doubles, so 4 million cells at 2000 points
    would need about 18 GB.  Chunking the targets keeps that allocation bounded.

    The kriging matrix depends only on the observations, so it is identical in
    every chunk and the only thing that changes is how many right-hand sides
    LAPACK is handed at once.  That does shift the last bits: a chunked run
    matches an unchunked one to about 1e-14 relative, which is the same
    difference you get by calling ``krige`` on one target instead of all of
    them.  With a single chunk the two are bit-for-bit equal.  The fields this
    feeds are Float32 rasters, so that margin is far below what the output can
    represent.

    ``progress`` is called as ``progress(done, total)`` after each chunk, which
    the Processing algorithm uses to report and to honour cancellation
    (returning ``False`` from it stops the run and raises ``RuntimeError``).
    """
    t = np.atleast_2d(np.asarray(targets, dtype=float))
    m = len(t)
    if m == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty

    n = len(np.asarray(values, dtype=float))
    p = 0 if drift == "none" else _drift_basis(np.asarray(coords, dtype=float),
                                               drift).shape[1]
    rows = n + p
    # Chunk so that one right-hand side stays under the byte budget, but never
    # below a single target.
    per_chunk = max(1, int(max_rhs_bytes // max(1, rows * 8)))

    pred_out = np.empty(m, dtype=float)
    var_out = np.empty(m, dtype=float)
    for start in range(0, m, per_chunk):
        stop = min(start + per_chunk, m)
        pred, var = krige(coords, values, t[start:stop], model, nugget, psill,
                          rng, drift, alpha)
        pred_out[start:stop] = pred
        var_out[start:stop] = var
        if progress is not None and progress(stop, m) is False:
            raise RuntimeError("Kriging cancelled.")
    return pred_out, var_out


def loo_cross_validation(coords, values, model, nugget, psill, rng,
                         drift="none", alpha=1.5):
    """Leave-one-out kriging validation.

    Every point is predicted from all the others, so the metrics are an
    honest statement about the fitted model rather than about the fit to the
    data it was fitted from.  ``mse`` near 0 and ``rmsse`` near 1 mean the
    variogram is neither over- nor under-stating the prediction error --
    kriging's standard validation test.

    Costs n kriging solves, hence ``LOO_SOFT_LIMIT``.
    """
    c = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    n = len(z)
    if n < 4:
        raise ValueError("Leave-one-out validation needs at least 4 points.")
    err = np.zeros(n)
    sd = np.zeros(n)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        pred, var = krige(c[keep], z[keep], c[i:i + 1], model, nugget, psill,
                          rng, drift, alpha)
        err[i] = z[i] - pred[0]
        sd[i] = math.sqrt(max(float(var[0]), 1e-300))
    std = err / sd
    return {
        "n": int(n),
        "me": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mse": float(std.mean()),
        "rmsse": float(np.sqrt(np.mean(std ** 2))),
        "errors": err,
        "standardized": std,
    }


def quality_flags(fit, loo=None, max_lag=None):
    """Plain-language warnings a planning analyst can act on."""
    flags = []
    r2 = fit.get("r2")
    if r2 is not None and np.isfinite(r2):
        if r2 < 0.5:
            flags.append(
                f"The fitted model explains little of the empirical "
                f"variogram (R2 = {r2:.2f}). The field may have no spatial "
                "structure at this scale, or the lag bins may be too few."
            )
    # A range at or past the last distance bin means the curve had not levelled
    # off, so the range is describing the extent of the data rather than a
    # property of the field.  An unremoved trend is the usual cause.
    rng_fit = fit.get("range")
    if (max_lag and rng_fit is not None and np.isfinite(rng_fit)
            and rng_fit >= 0.9 * float(max_lag)):
        flags.append(
            f"The fitted range ({rng_fit:.6g}) reaches the largest distance "
            f"binned ({float(max_lag):.6g}), so the variogram was still rising "
            "at the last bin and never levelled off. The range is describing "
            "how far the data extends rather than a real limit to spatial "
            "similarity. The usual cause is a trend left in the values; "
            "removing one, or binning over a shorter distance, is the way to "
            "tell the two apart."
        )
    if fit.get("nugget", 0.0) > 0.0 and fit.get("sill", 0.0) > 0.0:
        ratio = fit["nugget"] / fit["sill"]
        if ratio > 0.8:
            flags.append(
                f"The nugget is {ratio:.0%} of the sill, so most of the "
                "variation is at a scale finer than the sampling. Predictions "
                "will be close to a local mean and the surface will look "
                "smooth but carry little local information."
            )
    if loo is not None:
        if np.isfinite(loo.get("mse", float("nan"))) and abs(loo["mse"]) > 0.4:
            flags.append(
                f"Leave-one-out mean standardised error is {loo['mse']:+.2f}, "
                "far from 0. Predictions are systematically biased, which "
                "usually means the drift is mis-specified for this field."
            )
        rmsse = loo.get("rmsse", float("nan"))
        if np.isfinite(rmsse) and (rmsse < 0.7 or rmsse > 1.4):
            direction = "under" if rmsse > 1 else "over"
            flags.append(
                f"Leave-one-out RMS standardised error is {rmsse:.2f}, so the "
                f"kriging variance {direction}-states the real prediction "
                "error. Treat the variance surface as relative, not absolute."
            )
    return flags
