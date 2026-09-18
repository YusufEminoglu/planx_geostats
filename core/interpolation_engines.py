# -*- coding: utf-8 -*-
"""Deterministic interpolators and the cross-validation protocol they share.

Kriging in ``geostat_engines`` starts from a model of the field's spatial
structure.  The interpolators here start from nothing but the observations
themselves: a distance rule, a kernel, or a fitted polynomial.  That makes them
cheaper and easier to explain, and it is also their limitation - none of them
carries a model of how the field varies, so none of them can report a
prediction variance.  Their only honest quality measure is how well they
predict points they were not fitted to, which is why cross-validation lives in
this module beside them rather than inside each tool.

numpy-only, matching the rule the rest of ``core`` already follows:
``ml_engines`` hand-rolls a Wasserstein distance rather than import scipy, and
``stats_engines`` hand-rolls a chi-square survival function for the same
reason.  scipy is present in both QGIS runtimes only because scikit-learn pulls
it in, which is not a dependency this plugin declares, so building on it would
produce a tool that fails in an environment where every declared package is
installed.
"""
from __future__ import annotations

import numpy as np


#: Search neighbourhoods.  ``all`` uses every observation, ``nearest`` the k
#: closest, ``radius`` every observation inside a distance.  The distinction
#: matters more than it looks: with ``all`` a single distant point still pulls
#: on every cell, which is why the local search options exist at all.
NEIGHBOURHOODS = ("all", "nearest", "radius")

#: Radial basis kernels, by their conventional names.  Only ``thin_plate`` is
#: conditionally positive definite of order 2, so only it needs the polynomial
#: term that makes its system solvable - see ``rbf_weights``.
RBF_KERNELS = ("thin_plate", "multiquadric", "inverse_multiquadric", "gaussian")

#: Kernels whose shape parameter has to be chosen.  Thin-plate has none: it is
#: scale-invariant, so a shape parameter for it would be meaningless rather
#: than merely redundant.
RBF_SHAPED_KERNELS = tuple(k for k in RBF_KERNELS if k != "thin_plate")

#: Polynomial trend orders, and the number of coefficients each one fits.  The
#: term count is also the smallest observation count the order can be fitted
#: from, which is why it is a dict rather than arithmetic in two places.
TREND_ORDERS = (1, 2, 3)
TREND_TERM_COUNTS = {1: 3, 2: 6, 3: 10}

#: Observation counts above which the dominant cost stops being negligible, so
#: the tool should say the run will not be instant rather than leaving the user
#: to discover it.  Not hard limits - the caller sets its own.
IDW_SOFT_LIMIT = 20_000
RBF_SOFT_LIMIT = 4_000
TREND_SOFT_LIMIT = 20_000

#: One exact leave-one-out refit for a radial basis function is one dense
#: solve, so leave-one-out is affordable only on a subset.  Above this count
#: the tools switch to k-fold *and say so*, because the two protocols are not
#: comparable - k-fold fits each model on a smaller sample, so its error is
#: systematically larger - and reporting one as the other is the failure mode
#: this constant exists to prevent.
RBF_LOO_LIMIT = 400

#: Folds used when leave-one-out is not affordable.  Ten is the conventional
#: compromise between variance and the cost of ten refits.
DEFAULT_FOLDS = 10

#: Ceiling on one (observations x targets-in-chunk) distance block, so a large
#: grid does not materialise 80 million distances at once.
DISTANCE_BYTE_BUDGET = 64 * 1024 * 1024

#: Distance below which a target counts as sitting on an observation.  Exact
#: arithmetic would compare against zero, but coordinates arrive as doubles
#: from a GIS, so a cell centre meant to coincide with a point differs in the
#: last bits.
COINCIDENT = 1e-9


# ── shared geometry ──────────────────────────────────────────────────────────

def pair_dist(a, b):
    """Euclidean distance matrix between two ``(n, 2)`` coordinate arrays."""
    return np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])


def iter_chunks(n_targets, n_obs, budget=DISTANCE_BYTE_BUDGET):
    """``(start, stop)`` ranges keeping one distance block under ``budget``.

    Every interpolator here evaluates a whole grid against every observation,
    so a 200 x 200 surface over 2000 points is 80 million distances, or 640 MB
    in one array.  Chunking the targets is what keeps the peak bounded whatever
    the grid and point count.  Yields nothing for a zero-length target set.
    """
    if n_targets <= 0:
        return
    per = max(1, int(budget // max(1, int(n_obs) * 8)))
    for start in range(0, int(n_targets), per):
        yield start, min(start + per, int(n_targets))


def as_points(coords):
    """Validate and return an ``(n, 2)`` float array."""
    c = np.asarray(coords, dtype=float)
    if c.size == 0:
        return c.reshape(0, 2)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"Coordinates must be an (n, 2) array; got shape {c.shape}.")
    if not np.all(np.isfinite(c)):
        raise ValueError("Coordinates contain a non-finite value.")
    return c


def as_values(values):
    """Validate and return a 1-D float array."""
    z = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(z)):
        raise ValueError("Values contain a non-finite entry.")
    return z


def mean_nearest_distance(coords):
    """Mean distance from an observation to its closest neighbour.

    The length scale the shaped radial basis kernels are set from by default.
    Deriving it from the data keeps the default usable without asking for a
    parameter most users have no basis to choose; it stays exposed for anyone
    who wants to set it deliberately.
    """
    c = as_points(coords)
    n = len(c)
    if n < 2:
        return 0.0
    total = 0.0
    for start, stop in iter_chunks(n, n):
        block = pair_dist(c[start:stop], c)
        block[np.arange(stop - start), np.arange(start, stop)] = np.inf
        total += float(block.min(axis=1).sum())
    return total / n


# ── convex hull, for saying where a surface is extrapolation ─────────────────

def convex_hull(points):
    """Hull vertices, counter-clockwise, with collinear runs removed.

    Andrew's monotone chain.  Written here rather than reached for because its
    only consumer is a mask saying which cells of a surface lie outside the
    observations - and for that the hull is the whole story, since every
    interpolator in this module answers a query outside it by extending its own
    rule rather than by reporting that it has no data.

    Returns fewer than 3 vertices when the observations are collinear or
    coincident, which callers must treat as "outside has no meaning".
    """
    p = as_points(points)
    if len(p) == 0:
        return p
    pts = p[np.lexsort((p[:, 1], p[:, 0]))]
    if len(pts) > 1:
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(np.diff(pts, axis=0) != 0.0, axis=1)
        pts = pts[keep]
    if len(pts) < 3:
        return pts

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for pt in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def within_hull(targets, hull):
    """Boolean mask of targets inside a convex hull, boundary included.

    Vectorised over the targets: the loop runs once per hull vertex, not once
    per cell, so a 40 000-cell grid over a 50-vertex hull is 50 array passes.
    The tolerance is scaled to the hull's own size, so a cell centre landing
    exactly on an edge is not excluded by the last bit of its coordinate.
    """
    t = np.atleast_2d(as_points(targets))
    h = np.asarray(hull, dtype=float)
    if len(h) < 3 or len(t) == 0:
        return np.zeros(len(t), dtype=bool)
    span = float(np.max(h.max(axis=0) - h.min(axis=0)))
    tol = max(1e-12, 1e-10 * span * span)
    inside = np.ones(len(t), dtype=bool)
    n = len(h)
    for i in range(n):
        a = h[i]
        edge = h[(i + 1) % n] - a
        rel = t - a
        inside &= (edge[0] * rel[:, 1] - edge[1] * rel[:, 0]) >= -tol
    return inside


def extrapolation_summary(coords, targets):
    """How much of a target grid falls outside the observations' hull.

    Every tool in this group can be pointed at a grid larger than its data, and
    none of them refuses: they extend their rule into empty space and return
    numbers that nothing constrains.  This turns that into a statement the
    report can make instead of a surprise on the map.

    Returns ``None`` when the hull is degenerate, because then "inside" has no
    meaning to report - and an absent summary is honest, whereas a zero would
    read as "nothing is extrapolated".
    """
    hull = convex_hull(coords)
    if len(hull) < 3:
        return None
    t = np.atleast_2d(as_points(targets))
    inside = within_hull(t, hull)
    total = int(inside.size)
    outside = total - int(inside.sum())
    return {
        "total": total,
        "outside": outside,
        "fraction": (outside / total) if total else 0.0,
        "hull_vertices": int(len(hull)),
        "inside_mask": inside,
    }


# ── search neighbourhoods ────────────────────────────────────────────────────

def _neighbour_split(dist, k, radius):
    """``(indices, distances, valid)`` for one block of targets.

    All three are ``(chunk, k_eff)`` and ordered nearest-first.  A masked entry
    is a neighbour the caller must not use.  Keeping the shape rectangular and
    masking afterwards avoids a ragged array per target, which on a
    40 000-cell grid would cost more than the interpolation itself.
    """
    chunk, n = dist.shape
    k_eff = n if k is None else min(int(k), n)
    if k_eff >= n:
        idx = np.broadcast_to(np.arange(n), (chunk, n)).copy()
    else:
        idx = np.argpartition(dist, k_eff - 1, axis=1)[:, :k_eff]
        picked = np.take_along_axis(dist, idx, axis=1)
        idx = np.take_along_axis(idx, np.argsort(picked, axis=1, kind="stable"),
                                 axis=1)
    picked = np.take_along_axis(dist, idx, axis=1)
    valid = np.ones(picked.shape, dtype=bool)
    if radius is not None:
        valid &= picked <= float(radius)
    return idx, picked, valid


# ── inverse distance weighting ───────────────────────────────────────────────

def idw(coords, values, targets, power=2.0, k=None, radius=None, progress=None):
    """Inverse distance weighted interpolation.

    Each target is a weighted mean of the observations, weighted by
    ``1 / d ** power``.  ``power`` is the whole of the method: at 1 the surface
    is close to a nearest-neighbour map, at 2 it is the conventional choice,
    and past about 4 it is so dominated by the nearest few points that the rest
    of the dataset stops mattering and the surface develops the local
    bull's-eyes the method is criticised for.

    A target sitting on an observation returns that observation's value, which
    is the limit of the weight function rather than a special case invented
    here.  A target with no neighbour inside the search radius returns NaN,
    which the caller writes as no-data: there is no value to report, and
    returning a number anyway is how a radius search silently becomes a
    different interpolator.

    Returns ``(prediction, used, exact_hits)``: the neighbour count actually
    used per target, and how many targets landed on an observation.
    """
    c = as_points(coords)
    z = as_values(values)
    t = np.atleast_2d(as_points(targets))
    n = len(z)
    if n < 1:
        raise ValueError("Inverse distance weighting needs at least 1 observation.")
    p = float(power)
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError(f"Power must be a positive number; got {power}.")
    if k is not None and int(k) < 1:
        raise ValueError(f"The neighbour count must be at least 1; got {k}.")
    if radius is not None and not float(radius) > 0.0:
        raise ValueError(f"The search radius must be positive; got {radius}.")

    pred = np.full(len(t), np.nan, dtype=float)
    used = np.zeros(len(t), dtype=int)
    exact = 0
    for start, stop in iter_chunks(len(t), n):
        block = t[start:stop]
        dist = pair_dist(c, block).T
        idx, picked, valid = _neighbour_split(dist, k, radius)
        hit_row = (valid & (picked <= COINCIDENT)).any(axis=1)

        safe = np.where(picked > 0.0, picked, 1.0)
        weight = np.where(valid, safe ** (-p), 0.0)
        total = weight.sum(axis=1)
        block_pred = np.full(len(block), np.nan)
        ok = total > 0.0
        if ok.any():
            block_pred[ok] = (weight * z[idx]).sum(axis=1)[ok] / total[ok]
        if hit_row.any():
            first = np.argmax(valid & (picked <= COINCIDENT), axis=1)
            hit_idx = np.take_along_axis(idx, first[:, None], axis=1)[:, 0]
            block_pred[hit_row] = z[hit_idx[hit_row]]
        pred[start:stop] = block_pred
        used[start:stop] = valid.sum(axis=1)
        exact += int(hit_row.sum())
        if progress is not None and progress(stop, len(t)) is False:
            raise RuntimeError("Interpolation cancelled.")
    return pred, used, exact


# ── radial basis functions ───────────────────────────────────────────────────

def rbf_kernel(r, kernel, shape):
    """Evaluate a radial basis kernel on a distance array."""
    d = np.asarray(r, dtype=float)
    if kernel == "thin_plate":
        out = np.zeros_like(d)
        nz = d > 0.0
        out[nz] = d[nz] ** 2 * np.log(d[nz])
        return out
    s = float(shape)
    if kernel == "multiquadric":
        return np.sqrt(d * d + s * s)
    if kernel == "inverse_multiquadric":
        return 1.0 / np.sqrt(d * d + s * s)
    if kernel == "gaussian":
        return np.exp(-(d * d) / (s * s))
    raise ValueError(
        f"Unknown radial basis kernel '{kernel}'. Expected one of: "
        f"{', '.join(RBF_KERNELS)}."
    )


def rbf_weights(coords, values, kernel="thin_plate", shape=None, smoothing=0.0):
    """Solve the radial basis system for one dataset.

    Thin-plate is the one kernel here that is *conditionally* positive
    definite: its system is solvable only with a linear polynomial term and the
    side conditions that go with it.  Dropping that term still leaves a matrix
    that inverts, so the mistake is invisible - the surface still passes through
    every observation and looks plausible - while being a different estimator
    with worse behaviour away from the data.  The polynomial is therefore part
    of the thin-plate branch rather than an option.

    ``smoothing`` adds a ridge to the kernel matrix, trading exactness for
    resistance to noisy observations.  With zero smoothing, the default, the
    interpolation is exact and the reported residual is zero to rounding; with
    smoothing it is not, and the residual is the honest statement of how much
    exactness was given up.
    """
    c = as_points(coords)
    z = as_values(values)
    n = len(z)
    if n < 3:
        raise ValueError("Radial basis interpolation needs at least 3 observations.")
    if kernel not in RBF_KERNELS:
        raise ValueError(
            f"Unknown radial basis kernel '{kernel}'. Expected one of: "
            f"{', '.join(RBF_KERNELS)}."
        )
    lam = float(smoothing)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError(f"Smoothing must be zero or positive; got {smoothing}.")

    if kernel == "thin_plate":
        resolved_shape = None
    else:
        resolved_shape = (mean_nearest_distance(c) if shape is None
                          else float(shape))
        if not np.isfinite(resolved_shape) or resolved_shape <= 0.0:
            raise ValueError(
                "The kernel's shape parameter must be a positive distance; got "
                f"{shape}. Leave it unset to derive it from the observation "
                "spacing."
            )

    kernel_matrix = rbf_kernel(pair_dist(c, c), kernel, resolved_shape)
    augmented = kernel == "thin_plate"
    polynomial = (np.column_stack([np.ones(n), c[:, 0], c[:, 1]])
                  if augmented else None)
    size = n + (3 if augmented else 0)

    matrix = np.zeros((size, size))
    matrix[:n, :n] = kernel_matrix
    if augmented:
        matrix[:n, n:] = polynomial
        matrix[n:, :n] = polynomial.T
    matrix[:n, :n] += lam * np.eye(n)

    rhs = np.zeros(size)
    rhs[:n] = z
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        if augmented:
            raise ValueError(
                "The thin-plate system is singular, which means the "
                "observations are collinear - all of them on one straight "
                "line, leaving the polynomial term undetermined. Add an "
                "observation off that line, or use a multiquadric kernel, "
                "which does not need the polynomial term."
            ) from exc
        raise ValueError(
            "The radial basis system is singular, which means two or more "
            "observations sit at the same coordinates. Remove the duplicates, "
            "or raise the smoothing parameter to make the system solvable."
        ) from exc

    weights = solution[:n]
    poly = solution[n:] if augmented else np.zeros(0)
    fitted = kernel_matrix @ weights
    if augmented:
        fitted = fitted + polynomial @ poly
    residual = z - fitted
    return {
        "n": int(n),
        "kernel": kernel,
        "shape": resolved_shape,
        "smoothing": lam,
        "weights": weights,
        "poly": poly,
        "augmented": augmented,
        "rms_residual": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def rbf_predict(state, coords, targets):
    """Evaluate a solved radial basis system at new locations."""
    c = as_points(coords)
    t = np.atleast_2d(as_points(targets))
    if len(t) == 0:
        return np.zeros(0)
    out = rbf_kernel(pair_dist(c, t).T, state["kernel"],
                     state["shape"]) @ state["weights"]
    if state["augmented"]:
        poly = state["poly"]
        out = out + poly[0] + poly[1] * t[:, 0] + poly[2] * t[:, 1]
    return out


# ── polynomial trend surface ─────────────────────────────────────────────────

def normalise_coords(coords, centre=None, scale=None):
    """Centre and scale coordinates so a polynomial basis stays conditioned.

    A trend surface is fitted on projected coordinates, which are typically in
    the hundreds of thousands of metres.  Building ``x^3`` from those gives
    columns spanning nine orders of magnitude, and forming the normal equations
    squares the condition number on top of that - so the matrix is numerically
    singular long before it is mathematically singular.  The symptom is not a
    wrong number: it is a rank test that reports a perfectly determined cubic
    as under-determined, and local fits that refuse to happen at all.  Measured
    on a 10 x 10 grid before this was added, a second-order local trend fitted
    only 35 of 81 interior cells; afterwards, all 81.

    Returns ``(normalised, centre, scale)``.  The pair is carried in the fitted
    state so predictions are evaluated in the same parametrisation, which is
    what makes a coefficient mean the same thing at fit and at predict time.
    Coefficients are reported in these units, with the centring and scaling
    stated - the alternative, expanding the polynomial back to raw coordinates,
    replaces a stated normalisation with a silent one.
    """
    c = as_points(coords)
    if centre is None or scale is None:
        if len(c) == 0:
            return c, np.zeros(2), 1.0
        low = c.min(axis=0)
        high = c.max(axis=0)
        centre = (low + high) / 2.0
        scale = float(np.max(high - low)) / 2.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
    return (c - np.asarray(centre, dtype=float)) / float(scale), \
        np.asarray(centre, dtype=float), float(scale)


def trend_basis(coords, order):
    """Polynomial basis columns, in the order the coefficients are reported.

    Expects coordinates already centred and scaled by ``normalise_coords``;
    callers that go straight to this with raw map coordinates are building an
    ill-conditioned system.
    """
    c = as_points(coords)
    o = int(order)
    if o not in TREND_TERM_COUNTS:
        raise ValueError(
            f"Unknown trend order {order}. Expected one of: "
            f"{', '.join(str(v) for v in TREND_ORDERS)}."
        )
    x = c[:, 0]
    y = c[:, 1]
    columns = [np.ones(len(c)), x, y]
    if o >= 2:
        columns += [x * x, y * y, x * y]
    if o >= 3:
        columns += [x ** 3, x * x * y, x * y * y, y ** 3]
    return np.column_stack(columns)


def trend_terms(order):
    """Coefficient names, matching the columns ``trend_basis`` builds."""
    o = int(order)
    if o == 1:
        return ("intercept", "x", "y")
    if o == 2:
        return ("intercept", "x", "y", "x^2", "y^2", "x*y")
    if o == 3:
        return ("intercept", "x", "y", "x^2", "y^2", "x*y",
                "x^3", "x^2*y", "x*y^2", "y^3")
    raise ValueError(
        f"Unknown trend order {order}. Expected one of: "
        f"{', '.join(str(v) for v in TREND_ORDERS)}."
    )


def _least_squares(design, z):
    """Solve ``design @ c = z`` and return ``(c, inverse_normal, rank)``.

    A rank-deficient design is reported rather than silently tolerated: a
    cubic surface over observations that all sit on one line has infinitely
    many solutions, all of them meaningless, and the user needs to hear that
    rather than see whichever one the solver picked.
    """
    normal = design.T @ design
    # The rank test runs before the inverse, because inverting a rank-deficient
    # normal matrix raises a bare "singular matrix" from LAPACK - which is true
    # but says nothing about why.  A rank test distinguishes the two real
    # causes: too few directions in the observation locations, or a design that
    # only looks adequate.
    rank = int(np.linalg.matrix_rank(normal))
    if rank < design.shape[1]:
        raise ValueError(
            f"The trend surface is under-determined: this polynomial order "
            f"needs {design.shape[1]} independent directions in the "
            f"observation locations and only {rank} are available. Lower the "
            "order, or widen the search neighbourhood."
        )
    try:
        inverse = np.linalg.inv(normal)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - rank passed
        raise ValueError(
            "The trend surface could not be solved even though its design has "
            "full rank, which means the coordinates are numerically "
            "degenerate. Check the input layer's coordinates for values that "
            "are almost but not quite coincident."
        ) from exc
    return inverse @ (design.T @ z), inverse, rank


def trend_fit(coords, values, order=1, centre=None, scale=None):
    """Fit a global polynomial trend surface by least squares.

    The fit happens in centred and scaled coordinates (see
    ``normalise_coords``), so the coefficients are in those units and the
    centring and scaling are returned alongside them rather than left implicit.
    A local fit should pass the centre and scale of the whole dataset, so that
    a coefficient from one window is comparable with the same coefficient from
    the next - otherwise every window re-centres on itself and the coefficient
    surface becomes unreadable.
    """
    c = as_points(coords)
    z = as_values(values)
    o = int(order)
    terms = TREND_TERM_COUNTS.get(o)
    if terms is None:
        raise ValueError(
            f"Unknown trend order {order}. Expected one of: "
            f"{', '.join(str(v) for v in TREND_ORDERS)}."
        )
    if len(z) <= terms:
        raise ValueError(
            f"A trend of order {o} has {terms} coefficients, so it needs at "
            f"least {terms + 1} observations; got {len(z)}. Use a lower order."
        )
    normalised, used_centre, used_scale = normalise_coords(c, centre, scale)
    design = trend_basis(normalised, o)
    coefficients, inverse, rank = _least_squares(design, z)
    fitted = design @ coefficients
    residual = z - fitted
    dof = len(z) - terms
    sigma2 = float(residual @ residual / dof)
    total = float(((z - z.mean()) ** 2).sum())
    return {
        "order": o,
        "coefficients": coefficients,
        "inverse_normal": inverse,
        "rank": rank,
        "centre": used_centre,
        "scale": used_scale,
        "sigma2": sigma2,
        "residual_std": float(np.sqrt(sigma2)),
        "fitted": fitted,
        "residual": residual,
        "r2": (1.0 - float(residual @ residual) / total) if total > 0 else None,
        "n": int(len(z)),
    }


def trend_predict(state, targets, se=False):
    """Evaluate a fitted trend surface, optionally with its standard error.

    The standard error is the ordinary regression one,
    ``sigma^2 * x0' (X'X)^-1 x0``.  It describes how well the polynomial's
    coefficients are determined; it does **not** include any spatial
    correlation, so it understates the true error of the surface between
    observations and grows without bound outside them.  Reported for what it is
    - the first term of the real uncertainty, not the whole of it.
    """
    t = np.atleast_2d(as_points(targets))
    normalised, _, _ = normalise_coords(t, state["centre"], state["scale"])
    design = trend_basis(normalised, state["order"])
    pred = design @ state["coefficients"]
    if not se:
        return pred
    leverage = np.einsum("ij,jk,ik->i", design, state["inverse_normal"], design)
    return pred, np.sqrt(np.maximum(state["sigma2"] * leverage, 0.0))


def trend_surface(coords, values, targets, order=1, k=None, radius=None,
                  progress=None):
    """Global or moving-window polynomial trend surface.

    With no search neighbourhood this is a single least-squares fit over the
    whole dataset - ArcGIS's global polynomial interpolation.  With one, the
    polynomial is refitted inside each target's own neighbourhood, which can
    follow a field that a single global polynomial would smooth away, at the
    cost of a fit per cell.

    Returns ``(prediction, standard_error, used)``.  ``used`` is the
    observation count behind each cell: constant for the global fit, and the
    neighbourhood size for the local one, which is where a cell with too few
    neighbours to fit the order shows up as NaN rather than as a fit from three
    points.
    """
    c = as_points(coords)
    z = as_values(values)
    t = np.atleast_2d(as_points(targets))
    o = int(order)
    terms = TREND_TERM_COUNTS.get(o)
    if terms is None:
        raise ValueError(
            f"Unknown trend order {order}. Expected one of: "
            f"{', '.join(str(v) for v in TREND_ORDERS)}."
        )

    pred = np.full(len(t), np.nan, dtype=float)
    se = np.full(len(t), np.nan, dtype=float)
    used = np.zeros(len(t), dtype=int)

    # One centring and scaling for the whole run, taken from every observation
    # rather than recomputed per window.  A window that re-centred on itself
    # would give each cell coefficients in a different parametrisation, so the
    # "coefficient surface" would be meaningless and a cell's fit would depend
    # on where its neighbours happen to sit rather than on the field.
    _, centre, scale = normalise_coords(c)

    if k is None and radius is None:
        state = trend_fit(c, z, o, centre, scale)
        pred, se = trend_predict(state, t, se=True)
        used[:] = len(z)
        return pred, se, used

    for start, stop in iter_chunks(len(t), len(z)):
        block = t[start:stop]
        idx, picked, valid = _neighbour_split(pair_dist(c, block).T, k, radius)
        for row in range(len(block)):
            sel = idx[row][valid[row]]
            used[start + row] = int(len(sel))
            if len(sel) <= terms:
                continue
            try:
                local = trend_fit(c[sel], z[sel], o, centre, scale)
            except ValueError:
                # A neighbourhood that cannot pin the order is left as no-data
                # rather than answered with a lower order the user did not ask
                # for; the count in `used` says which cells these are.
                continue
            point = block[row:row + 1]
            value, error = trend_predict(local, point, se=True)
            pred[start + row] = float(value[0])
            se[start + row] = float(error[0])
        if progress is not None and progress(stop, len(t)) is False:
            raise RuntimeError("Interpolation cancelled.")
    return pred, se, used


# ── cross-validation ─────────────────────────────────────────────────────────

def as_prediction(out, expected, fold_index):
    """A predictor's return value as one float per test location.

    The contract is that ``predict`` returns predictions and nothing else, but
    an engine that returns a tuple - :func:`idw` returns ``(prediction, used,
    exact_hits)`` - reaches numpy as a ragged three-item sequence and raises
    "setting an array element with a sequence", which names neither the
    predictor nor the fold nor the real mistake.  A predictor that hands back
    the engine's whole return value is a bug in the caller, and it is worth a
    message that says so: the failure it produces is otherwise reported as a
    method that could not be scored, which reads like a property of the data.
    """
    try:
        array = np.asarray(out, dtype=float)
    except ValueError as exc:
        raise ValueError(
            f"The interpolator returned {type(out).__name__} rather than one "
            f"value per test location ({exc})."
        ) from exc
    if array.ndim != 1 or array.size != expected:
        raise ValueError(
            f"The interpolator returned {array.shape} for {expected} test "
            f"location(s) in fold {fold_index + 1}; it must return one "
            "prediction per location."
        )
    return array


def cross_validate(predict, coords, values, protocol="loo", folds=DEFAULT_FOLDS,
                   seed=0, progress=None):
    """Score an interpolator on observations it was not fitted to.

    ``predict(train_coords, train_values, test_coords)`` must refit from
    scratch and return predictions at the test locations.  That contract is the
    point of the function: a deterministic interpolator like IDW or a radial
    basis function passes exactly through its own data, so a residual against
    the fitted surface is zero by construction and says nothing at all.  Only a
    refit can answer "how wrong is this where it has not been".

    ``protocol="loo"`` leaves one observation out at a time; ``"kfold"`` splits
    into ``folds`` groups.  The two are **not** interchangeable and the caller
    must report which one it used: k-fold fits each model on a fraction of the
    data, so its error is systematically larger, and a k-fold RMSE shown beside
    a leave-one-out RMSE reads as a worse method when it is only a stricter
    test.

    Returns the error statistics, the per-observation errors, and the protocol
    actually used.  The normalised RMSE is the one number comparable across
    fields, since RMSE alone is in the data's own units.
    """
    c = as_points(coords)
    z = as_values(values)
    n = len(z)
    if n < 4:
        raise ValueError(
            f"Cross-validation needs at least 4 observations; got {n}."
        )
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

    errors = np.full(n, np.nan, dtype=float)
    predicted = np.full(n, np.nan, dtype=float)
    for index, test in enumerate(groups):
        train = np.ones(n, dtype=bool)
        train[test] = False
        out = as_prediction(predict(c[train], z[train], c[test]), len(test),
                            index)
        predicted[test] = out
        errors[test] = z[test] - out
        if progress is not None and progress(index + 1, len(groups)) is False:
            raise RuntimeError("Cross-validation cancelled.")

    finite = np.isfinite(errors)
    if not finite.any():
        raise ValueError(
            "Cross-validation produced no usable predictions - every held-out "
            "observation had too few neighbours. Widen the search "
            "neighbourhood and try again."
        )
    if not finite.all():
        # Cells with too few neighbours are a real answer, not a failure, but
        # they must not silently leave NaNs propagating through the mean.
        dropped = int((~finite).sum())

        def _stat(value):
            return float(np.nanmean(value))
    else:
        dropped = 0

        def _stat(value):
            return float(np.mean(value))
    std = float(np.std(z))
    rmse = float(np.sqrt(np.mean(errors[finite] ** 2)))
    return {
        "protocol": "leave-one-out" if protocol == "loo" else f"{len(groups)}-fold",
        "protocol_id": protocol,
        "folds": int(len(groups)),
        "n": int(n),
        "n_scored": int(finite.sum()),
        "n_dropped": dropped,
        "me": _stat(errors),
        "rmse": rmse,
        "mae": _stat(np.abs(errors)),
        "max_abs_error": float(np.max(np.abs(errors[finite]))),
        "nrmse": (rmse / std) if std > 0 else None,
        "field_std": std,
        "errors": errors,
        "predicted": predicted,
    }


def cv_protocol_for(n_obs, loo_limit=None):
    """The most informative protocol affordable at this observation count.

    Leave-one-out whenever it can be paid for, because it is the strongest test
    and the only one whose result does not depend on a random split; k-fold
    above the limit, because the alternative there is no validation at all.
    Returned rather than decided inline so the caller can put the same answer
    in the report, the log and the layer attributes without three copies of the
    same rule.
    """
    if loo_limit is None or int(n_obs) <= int(loo_limit):
        return "loo", int(n_obs)
    return "kfold", DEFAULT_FOLDS


# ── natural neighbour ────────────────────────────────────────────────────────

#: Triangulating is O(n log n) in principle, but every insertion is located by
#: walking the mesh in Python, so the measured cost is closer to quadratic:
#: 500 points build in 0.10 s, 2 000 in 0.50 s, 8 000 in 2.9 s.  The ceiling
#: sits where that is still a wait rather than a refusal, and matches IDW's
#: because neither method has a dense solve to pay for.
NATURAL_SOFT_LIMIT = 20_000

#: Leave-one-out for this method means one re-triangulation per held-out
#: observation, because removing an observation changes the mesh rather than
#: only a right-hand side.  Measured here rather than guessed: 150 points take
#: 4 s, 250 take 12 s and 300 take 17.5 s, so the limit is set where the wait
#: is still worth the stronger score.  Above it k-fold validation runs instead
#: and the report names the protocol, because the two are not comparable.
NATURAL_LOO_LIMIT = 300

#: Relative slack in the in-circle test.  Ties are resolved *into* the cavity:
#: points on a common circle are common in gridded data, and a maximal cavity
#: keeps the boundary a simple polygon, whereas breaking the other way can
#: leave a cavity whose boundary is not star-shaped around the inserted point.
INCIRCLE_REL = 1e-12


def orient(a, b, c) -> float:
    """Twice the signed area of the triangle ``(a, b, c)``.

    Positive when the three points run counter-clockwise.  This is the whole of
    the inside/outside test for a counter-clockwise triangle: a point is inside
    exactly when it lies to the left of all three directed edges.
    """
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def circumcentres(a, b, c):
    """Circumcentres of a batch of point triples, each an ``(n, 2)`` array."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    c = np.atleast_2d(np.asarray(c, dtype=float))
    ax, ay = a[:, 0], a[:, 1]
    bx, by = b[:, 0], b[:, 1]
    cx, cy = c[:, 0], c[:, 1]
    den = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = ((ax * ax + ay * ay) * (by - cy)
              + (bx * bx + by * by) * (cy - ay)
              + (cx * cx + cy * cy) * (ay - by)) / den
        uy = ((ax * ax + ay * ay) * (cx - bx)
              + (bx * bx + by * by) * (ax - cx)
              + (cx * cx + cy * cy) * (bx - ax)) / den
    return np.column_stack([ux, uy])


class DelaunayTriangulation:
    """Incremental (Bowyer-Watson) Delaunay triangulation of 2-D points.

    Hand-written rather than taken from ``scipy.spatial`` for the reason the
    module docstring gives: scipy is not a dependency this plugin declares, so
    importing it here would create one tool that needs a package the installer
    never installs.  The scratch harness cross-checks the result against
    scipy's triangulation when scipy happens to be present; the shipped code
    never touches it.

    A super-triangle is built around the data and its three vertices are
    removed at the end, which is what lets incremental insertion treat the hull
    as ordinary interior.  Internal indices are therefore not the input
    indices: ``points`` is the input stacked on the three super vertices, and
    ``triangles`` is the filtered view of the real ones.
    """

    def __init__(self, points):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError("Triangulation needs an (n, 2) array of points.")
        self.n = len(p)
        if self.n < 3:
            raise ValueError("Triangulation needs at least three points.")

        lo, hi = p.min(axis=0), p.max(axis=0)
        centre = (lo + hi) / 2.0
        span = float(np.max(hi - lo))
        if span <= 0.0:
            raise ValueError(
                "Every observation shares one location, so there is no "
                "triangulation to build."
            )
        # Wide enough that its circumcircle contains the whole bounding box: at
        # 20 spans the circle reaches about 20.02 spans from the centre.
        super_tri = np.array([
            [centre[0] - 20.0 * span, centre[1] - span],
            [centre[0] + 20.0 * span, centre[1] - span],
            [centre[0], centre[1] + 20.0 * span],
        ])
        self.points = np.vstack([p, super_tri])

        self._triangles = [(self.n, self.n + 1, self.n + 2)]
        self._alive = [True]
        self._circles = [None]
        self._edge_tris = {}
        self._index_triangle(0)
        self._last = 0

        for k in range(self.n):
            self._insert(k)
        self._drop_super()

    # ── circle bookkeeping ───────────────────────────────────────────────────

    def _circle(self, t):
        if self._circles[t] is None:
            a, b, c = (self.points[i] for i in self._triangles[t])
            centre = circumcentres(a, b, c)[0]
            self._circles[t] = (centre, float(np.sum((a - centre) ** 2)))
        return self._circles[t]

    def _in_circle(self, t, p):
        centre, radius2 = self._circle(t)
        if not np.isfinite(radius2):
            return False
        d2 = float(np.sum((p - centre) ** 2))
        return d2 <= radius2 * (1.0 + INCIRCLE_REL)

    # ── graph bookkeeping ────────────────────────────────────────────────────

    def _directed_edges(self, t):
        a, b, c = self._triangles[t]
        return ((a, b), (b, c), (c, a))

    def _index_triangle(self, t):
        for a, b in self._directed_edges(t):
            self._edge_tris.setdefault((min(a, b), max(a, b)), set()).add(t)

    def _unindex_triangle(self, t):
        for a, b in self._directed_edges(t):
            key = (min(a, b), max(a, b))
            bucket = self._edge_tris.get(key)
            if bucket is not None:
                bucket.discard(t)
                if not bucket:
                    del self._edge_tris[key]

    def _other_triangle(self, a, b, t):
        """The alive triangle sharing edge ``(a, b)`` with ``t``, if any."""
        bucket = self._edge_tris.get((min(a, b), max(a, b)))
        if not bucket:
            return None
        for u in bucket:
            if u != t:
                return u
        return None

    def _add_triangle(self, verts):
        t = len(self._triangles)
        self._triangles.append(tuple(verts))
        self._alive.append(True)
        self._circles.append(None)
        self._index_triangle(t)
        return t

    def _retire(self, t):
        self._unindex_triangle(t)
        self._alive[t] = False
        self._circles[t] = None

    # ── insertion ────────────────────────────────────────────────────────────

    def _first_alive(self):
        for t in range(len(self._triangles)):
            if self._alive[t]:
                return t
        raise ValueError("The triangulation has no triangles left.")

    def _locate(self, p):
        """A triangle whose circumcircle contains ``p``, or ``None``.

        Walks from the last located triangle, crossing whichever edge ``p``
        lies to the right of.  A raster scanned row by row lands each cell near
        the previous one, so the walk is a handful of steps in practice; the
        scan at the end is the safety net for the case where the walk cannot
        leave the super-triangle.

        The walk must start on a live triangle.  Starting on a retired one is
        not a visible error at the start - it returns a triangle, and a
        triangle has the edges it always had - but its edges have been removed
        from the adjacency index, so its cavity comes back as itself alone and
        the retriangulation around it silently overlaps the triangles that
        replaced it.
        """
        t = self._last if self._alive[self._last] else self._first_alive()
        for _ in range(4 * len(self._triangles) + 16):
            tri = self._triangles[t]
            stepped = False
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                if orient(self.points[a], self.points[b], p) < 0.0:
                    nxt = self._other_triangle(a, b, t)
                    if nxt is None or not self._alive[nxt]:
                        stepped = False
                        break
                    t = nxt
                    stepped = True
                    break
            if not stepped:
                break
            self._last = t
        if self._in_circle(t, p):
            return t
        for u in range(len(self._triangles)):
            if self._alive[u] and self._in_circle(u, p):
                return u
        return None

    def _cavity(self, p, t0):
        """Every alive triangle whose circumcircle contains ``p``."""
        stack = [t0]
        seen = {t0}
        bad = []
        while stack:
            t = stack.pop()
            bad.append(t)
            for a, b in self._directed_edges(t):
                u = self._other_triangle(a, b, t)
                if u is None or u in seen or not self._alive[u]:
                    continue
                if self._in_circle(u, p):
                    seen.add(u)
                    stack.append(u)
        return bad

    def _cavity_boundary(self, bad):
        """The directed edges bounding a cavity, verified to form one cycle.

        The check is worth the loop.  A boundary that is not a single closed
        counter-clockwise cycle means the cavity was not star-shaped around the
        inserted point, and retriangulating it would produce overlapping
        triangles; raising here turns a silently corrupt triangulation into a
        message that says what happened.
        """
        members = set(bad)
        edges = []
        for t in bad:
            for a, b in self._directed_edges(t):
                u = self._other_triangle(a, b, t)
                if u is None or u not in members:
                    edges.append((a, b))
        tails, heads = {}, {}
        for a, b in edges:
            tails[a] = tails.get(a, 0) + 1
            heads[b] = heads.get(b, 0) + 1
        if set(tails) != set(heads) or any(v != 1 for v in tails.values()):
            raise ValueError(
                "The triangulation could not be updated: the cavity around a "
                "point did not close into a single polygon, which happens when "
                "observations are so nearly collinear or duplicated that the "
                "in-circle test cannot separate them. Remove duplicate "
                "locations or thin the observations."
            )
        return edges

    def _insert(self, k):
        p = self.points[k]
        t = self._locate(p)
        if t is None:
            raise ValueError(
                "A point could not be located inside the triangulation, which "
                "means the observation set is numerically degenerate."
            )
        bad = self._cavity(p, t)
        edges = self._cavity_boundary(bad)
        for u in bad:
            self._retire(u)
        for a, b in edges:
            self._add_triangle((a, b, k))

    def _drop_super(self):
        """Retire every triangle that touches a super-triangle vertex."""
        for t in range(len(self._triangles)):
            if self._alive[t] and max(self._triangles[t]) >= self.n:
                self._retire(t)

    # ── public view ──────────────────────────────────────────────────────────

    @property
    def triangles(self):
        """The real triangles, ``(m, 3)``, counter-clockwise, input indices."""
        return np.array(
            [self._triangles[t] for t in range(len(self._triangles))
             if self._alive[t] and max(self._triangles[t]) < self.n],
            dtype=int,
        )

    def cavity_boundary(self, p, t0):
        """The ordered neighbour ring around a target inside the triangulation.

        Returned as the vertex cycle rather than as edges, because the cycle is
        what the weights are defined on: consecutive ring vertices bound one
        Voronoi edge of the inserted point.
        """
        edges = self._cavity_boundary(self._cavity(p, t0))
        nxt = {}
        for a, b in edges:
            nxt[a] = b
        start = edges[0][0]
        ring = [start]
        while True:
            step = nxt.get(ring[-1])
            if step is None or step == start:
                break
            ring.append(step)
            if len(ring) > len(edges):
                return None
        if len(ring) != len(edges):
            return None
        return ring


def _bisector_half_plane(a, b):
    """``(nx, ny, c)`` for the half-plane ``{x : |x - a| <= |x - b|}``.

    ``|x-a|^2 <= |x-b|^2`` rearranges to ``2(b-a).x <= |b|^2 - |a|^2``, so the
    bisector is a straight line in the coordinates and the set closer to ``a``
    is one of its two sides.  That linearity is what makes the clipping below
    exact rather than iterative.
    """
    nx = 2.0 * (b[0] - a[0])
    ny = 2.0 * (b[1] - a[1])
    return nx, ny, b[0] * b[0] + b[1] * b[1] - a[0] * a[0] - a[1] * a[1]


def _clip_half_plane(poly, nx, ny, c):
    """Sutherland-Hodgman clip of a convex polygon to ``nx*x + ny*y <= c``."""
    out = []
    n = len(poly)
    for i in range(n):
        px, py = poly[i]
        qx, qy = poly[(i + 1) % n]
        dp = nx * px + ny * py - c
        dq = nx * qx + ny * qy - c
        if dp <= 0.0:
            out.append((px, py))
        if (dp < 0.0) != (dq < 0.0):
            t = dp / (dp - dq)
            out.append((px + t * (qx - px), py + t * (qy - py)))
    return out


def _polygon_area(poly):
    """Signed area of a polygon given as a list of ``(x, y)`` tuples."""
    total = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def natural_neighbour_weights(triangulation, target):
    """Sibson's natural-neighbour coordinates of one target.

    Returns ``(ring, weights)`` with the neighbours' indices and their
    normalised coordinates, ``(ring, None)`` when the geometry of the ring is
    degenerate, or ``None`` when the target has no neighbour ring at all.

    The weights are Sibson's (1981): the fraction of the target's new Voronoi
    cell that falls inside each natural neighbour's old one - the area that
    neighbour "loses" when the target is inserted.  ArcGIS's Natural Neighbor
    tool computes the same quantity, so the surface this feeds is the surface
    that tool draws.

    Two facts make this affordable without constructing a single Voronoi cell.
    The first is that the target's new cell is the polygon of circumcentres of
    the triangles around it, which the triangulation already has.  The second
    is Sibson's covering property: every point of the new cell has one of the
    *ring* vertices as its nearest old site, so "inside neighbour ``i``'s old
    cell" reduces to "closer to ring vertex ``i`` than to any other ring
    vertex".  The stolen area is therefore the new cell clipped by ``k - 1``
    bisector half-planes, each of which is a straight line.

    Sibson's coordinates reproduce a constant *and* a linear function exactly,
    which is what separates them from the cheaper Laplacian (non-Sibson)
    coordinates that take Voronoi edge *lengths* instead of areas.  The
    Laplacian form is non-negative and sums to one like this one, so it looks
    right on a smooth field, but it does not reproduce a plane - a fact worth
    recording because it was measured here, not assumed: on the harness's
    planar field it misses by about two observation spacings, which no
    interpolation scheme presented as natural-neighbour should do.
    """
    p = np.asarray(target, dtype=float)
    t0 = triangulation._locate(p)
    if t0 is None:
        return None
    raw = triangulation.cavity_boundary(p, t0)
    if raw is None or len(raw) < 3:
        return None
    ring = np.asarray(raw, dtype=int)

    k = len(ring)
    verts = triangulation.points[ring]
    # The target's own Voronoi cell: one vertex per ring edge, the circumcentre
    # of the triangle formed by the target and two consecutive ring vertices.
    a = np.broadcast_to(p, (k, 2))
    b = verts
    c = triangulation.points[np.roll(ring, -1)]
    centres = circumcentres(a, b, c)
    if not np.all(np.isfinite(centres)):
        return ring, None
    cell = [(float(x), float(y)) for x, y in centres]
    if _polygon_area(cell) < 0.0:
        cell.reverse()
    total_area = _polygon_area(cell)
    if not np.isfinite(total_area) or total_area <= 0.0:
        return ring, None

    stolen = []
    for i in range(k):
        poly = cell
        for j in range(k):
            if j == i:
                continue
            nx, ny, cc = _bisector_half_plane(verts[i], verts[j])
            poly = _clip_half_plane(poly, nx, ny, cc)
            if len(poly) < 3:
                break
        stolen.append(max(_polygon_area(poly), 0.0))
    total = float(sum(stolen))
    if not np.isfinite(total) or total <= 0.0:
        return ring, None
    return ring, np.asarray(stolen, dtype=float) / total


def natural_neighbour(coords, values, targets, progress=None):
    """Natural-neighbour interpolation over a target grid.

    Returns ``(prediction, used, detail)``: the interpolated values, the number
    of natural neighbours behind each cell, and a dict counting the cells that
    landed exactly on an observation and the cells whose ring geometry was
    degenerate.

    Cells outside the observations' convex hull are ``nan`` with ``used`` 0.
    That is not a policy the caller can switch off the way it is for the other
    methods: a natural neighbour is a vertex of the Voronoi cell the target
    would have, and a target outside the hull has no such cell, because its
    Voronoi region is unbounded in at least one direction.  There is nothing to
    mask away, since there is nothing computed to mask.

    ``used`` is a count of neighbours, not an uncertainty.  It is the closest
    thing this method offers to one and it is not one.
    """
    c = as_points(coords)
    z = as_values(values)
    t = np.atleast_2d(np.asarray(targets, dtype=float))
    m = len(t)
    if len(c) < 3:
        raise ValueError(
            "Natural neighbour interpolation needs at least 3 observations."
        )

    hull = convex_hull(c)
    if len(hull) < 3:
        raise ValueError(
            "Natural neighbour interpolation needs at least three observations "
            "that are not collinear. These lie on a line, so their Voronoi "
            "cells are unbounded strips rather than areas, and the natural "
            "neighbours of a point are not defined."
        )
    inside = within_hull(t, hull)
    triangulation = DelaunayTriangulation(c)

    prediction = np.full(m, np.nan)
    used = np.zeros(m, dtype=int)
    detail = {"exact": 0, "degenerate": 0, "no_ring": 0}

    for i in range(m):
        if not inside[i]:
            continue
        p = t[i]
        # A target on top of an observation is answered with it.  The weights
        # degenerate there - the target's Voronoi cell has zero area - and the
        # limit of the interpolant is the observed value, so this evaluates the
        # limit where it is not numerically reachable.
        same = np.flatnonzero(np.all(np.abs(c - p) <= COINCIDENT, axis=1))
        if len(same):
            prediction[i] = z[same[0]]
            used[i] = 1
            detail["exact"] += 1
            continue
        ring_weights = natural_neighbour_weights(triangulation, p)
        if ring_weights is None:
            detail["no_ring"] += 1
            continue
        ring, w = ring_weights
        if w is None:
            # Every Voronoi vertex of the target coincided, which happens when
            # the ring is numerically collinear with the target.  Inverse
            # squared distance over the same ring keeps the surface bounded by
            # the observations, and the count is reported rather than hidden.
            detail["degenerate"] += 1
            d = np.linalg.norm(c[ring] - p, axis=1)
            w = 1.0 / np.maximum(d, COINCIDENT) ** 2
            w = w / w.sum()
        prediction[i] = float(np.dot(w, z[ring]))
        used[i] = len(ring)
        if progress is not None and (i % 256 == 0 or i == m - 1):
            if progress(i + 1, m) is False:
                raise RuntimeError("Natural neighbour interpolation cancelled.")

    return prediction, used, detail
