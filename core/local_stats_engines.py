# -*- coding: utf-8 -*-
"""Rate smoothing and the extended LISA family (v3.11.0).

Everything here is numpy alone, like the rest of ``core``: the tools work
without PySAL installed, and the smoke suite exercises them without QGIS.
Where PySAL's ``esda`` implements the same statistic, the runtime matrix
compares the two on the bundled data (``tests/qgis_runtime_algorithm_matrix``
``_esda_cross_checks``), so "matches esda" is a measured statement.

Two conventions run through the module.

* Weights arrive as the ``neighbors``/``weights`` dicts every GeoStats tool
  builds with ``core.weights.build_weights_matrix`` and are turned into a
  :class:`SpatialGraph` - three aligned edge arrays - so lags are a single
  ``np.bincount`` instead of a Python loop per feature.
* Local inference is conditional permutation (Anselin 1995): the focal value
  is held fixed and its neighbour set is redrawn, without replacement, from
  the other n-1 observations.  Each focal unit gets its own random
  permutation of the others, so the draws of two units are independent.
  P-values are two-sided unless a statistic only has one meaningful tail,
  which the function says.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class SpatialGraph:
    """Sparse neighbour structure aligned to an observation array.

    ``focal[e]``/``neighbor[e]``/``weight[e]`` describe edge ``e``; edges are
    sorted by focal so the neighbours of ``i`` are the slice
    ``offsets[i]:offsets[i + 1]``.
    """

    def __init__(self, n: int, focal, neighbor, weight):
        focal = np.asarray(focal, dtype=np.int64)
        neighbor = np.asarray(neighbor, dtype=np.int64)
        weight = np.asarray(weight, dtype=float)
        order = np.lexsort((neighbor, focal))
        self.n = int(n)
        self.focal = focal[order]
        self.neighbor = neighbor[order]
        self.weight = weight[order]
        self.cardinality = np.bincount(self.focal, minlength=self.n)
        self.offsets = np.concatenate([[0], np.cumsum(self.cardinality)])

    @classmethod
    def from_dicts(cls, neighbors: dict, weights: dict, id_order) -> "SpatialGraph":
        """Build from GeoStats weight dicts, keeping only ids in ``id_order``.

        ``id_order`` is the list of feature ids whose values were usable, in
        the order of the value array; neighbours outside it (null values) are
        dropped, which is the same filtering every existing LISA tool applies.
        Self-links and duplicate links are removed here so a caller cannot
        double-count a neighbour.
        """
        index = {fid: i for i, fid in enumerate(id_order)}
        focal, neighbor, weight = [], [], []
        for i, fid in enumerate(id_order):
            seen = set()
            w_list = weights.get(fid, []) if weights else []
            for k, nid in enumerate(neighbors.get(fid, [])):
                j = index.get(nid)
                if j is None or j == i or j in seen:
                    continue
                seen.add(j)
                focal.append(i)
                neighbor.append(j)
                weight.append(float(w_list[k]) if k < len(w_list) else 1.0)
        return cls(len(id_order), focal, neighbor, weight)

    # -- transforms -------------------------------------------------------
    @property
    def rowsum(self) -> np.ndarray:
        return np.bincount(self.focal, weights=self.weight, minlength=self.n)

    def binary(self) -> "SpatialGraph":
        return SpatialGraph(self.n, self.focal, self.neighbor, np.ones_like(self.weight))

    def row_standardized(self) -> "SpatialGraph":
        rs = self.rowsum
        safe = np.where(rs[self.focal] > 0, rs[self.focal], 1.0)
        return SpatialGraph(self.n, self.focal, self.neighbor, self.weight / safe)

    def with_self(self, self_weight: float = 1.0) -> "SpatialGraph":
        """Add ``i -> i`` with ``self_weight`` (Gi*/Lee-style neighbourhoods)."""
        idx = np.arange(self.n)
        keep = self.focal != self.neighbor
        return SpatialGraph(
            self.n,
            np.concatenate([self.focal[keep], idx]),
            np.concatenate([self.neighbor[keep], idx]),
            np.concatenate([self.weight[keep], np.full(self.n, float(self_weight))]),
        )

    def symmetric_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Unique undirected pairs (i < j) - the contiguity edge set."""
        a = np.minimum(self.focal, self.neighbor)
        b = np.maximum(self.focal, self.neighbor)
        keep = a != b
        pairs = np.unique(np.stack([a[keep], b[keep]], axis=1), axis=0)
        if pairs.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return pairs[:, 0], pairs[:, 1]

    # -- algebra -----------------------------------------------------------
    def lag(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.bincount(
            self.focal, weights=self.weight * values[self.neighbor], minlength=self.n
        )

    def neighbors_of(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return self.neighbor[lo:hi], self.weight[lo:hi]

    @property
    def islands(self) -> np.ndarray:
        return np.flatnonzero(self.cardinality == 0)

    def components(self) -> np.ndarray:
        """Connected-component label per node (undirected)."""
        parent = np.arange(self.n)

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for a, b in zip(*self.symmetric_pairs()):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        roots = np.array([find(i) for i in range(self.n)])
        _, labels = np.unique(roots, return_inverse=True)
        return labels


# ---------------------------------------------------------------------------
# Permutation machinery
# ---------------------------------------------------------------------------


def conditional_draws(graph: SpatialGraph, permutations: int, seed: int):
    """Yield ``(i, draws, w)`` for every unit with at least one neighbour.

    ``draws`` is a ``(permutations, k_i)`` array of indices into the full
    observation array, each row a without-replacement sample of size k_i
    from the n-1 units other than ``i``; ``w`` is ``i``'s weight vector.
    """
    n = graph.n
    if n < 3 or permutations < 1:
        return
    rng = np.random.default_rng(int(seed))
    kmax = int(graph.cardinality.max()) if n else 0
    kmax = min(kmax, n - 1)
    if kmax <= 0:
        return
    # One block of ranks shared by all units, re-labelled for each unit
    # through its own random permutation of the others - the cost per unit
    # is O(n + P*k) rather than P full shuffles.
    ranks = np.argsort(rng.random((permutations, n - 1)), axis=1)[:, :kmax]
    for i in range(n):
        lo, hi = graph.offsets[i], graph.offsets[i + 1]
        k = int(hi - lo)
        if k == 0:
            continue
        k = min(k, n - 1)
        others = rng.permutation(n - 1)
        others = others + (others >= i)
        yield i, others[ranks[:, :k]], graph.weight[lo:lo + k]


def two_sided_p(observed: float, simulated: np.ndarray) -> float:
    """Two-sided permutation p-value, ``2 * (tail + 1) / (P + 1)``, capped at 1."""
    sims = np.asarray(simulated, dtype=float)
    sims = sims[np.isfinite(sims)]
    if sims.size == 0 or not np.isfinite(observed):
        return float("nan")
    tail = min(int(np.sum(sims >= observed)), int(np.sum(sims <= observed)))
    return float(min(1.0, 2.0 * (tail + 1.0) / (sims.size + 1.0)))


def greater_p(observed: float, simulated: np.ndarray) -> float:
    """One-sided (upper tail) permutation p-value, ``(#{sim >= obs} + 1) / (P + 1)``."""
    sims = np.asarray(simulated, dtype=float)
    if sims.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((np.sum(sims >= observed) + 1.0) / (sims.size + 1.0))


def _zscores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = values.std()
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(values)
    return (values - values.mean()) / sd


QUADRANT_CODES = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def quadrant_labels(left: np.ndarray, right: np.ndarray) -> list[str]:
    """HH/LH/LL/HL from the sign of a value and of its spatial lag."""
    out = []
    for a, b in zip(np.asarray(left, dtype=float), np.asarray(right, dtype=float)):
        if a >= 0 and b >= 0:
            out.append("HH")
        elif a < 0 and b >= 0:
            out.append("LH")
        elif a < 0 and b < 0:
            out.append("LL")
        else:
            out.append("HL")
    return out


# ---------------------------------------------------------------------------
# Special functions (no scipy dependency)
# ---------------------------------------------------------------------------


def _gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) (Numerical Recipes 6.2)."""
    if x <= 0:
        return 1.0
    if a <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, total, delta = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-14:
                break
        p = total * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - gln) * h))


def chi2_sf(x: float, dof: float) -> float:
    """Upper tail of a chi-square with possibly non-integer ``dof``."""
    if not (np.isfinite(x) and np.isfinite(dof)) or dof <= 0:
        return float("nan")
    return _gamma_q(dof / 2.0, x / 2.0)


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def _validate_rate_inputs(events, base):
    e = np.asarray(events, dtype=float)
    b = np.asarray(base, dtype=float)
    if e.shape != b.shape:
        raise ValueError("Events and base must have the same length.")
    if np.any(b <= 0):
        raise ValueError("Every base (population at risk) must be positive.")
    if np.any(e < 0):
        raise ValueError("Event counts cannot be negative.")
    return e, b


def raw_rate(events, base) -> np.ndarray:
    e, b = _validate_rate_inputs(events, base)
    return e / b


def global_eb_rate(events, base) -> dict:
    """Marshall (1991) global empirical Bayes rate.

    Each raw rate is shrunk towards the overall rate with weight
    ``w_i = A / (A + m / b_i)``, where ``m`` is the overall rate and ``A`` the
    method-of-moments between-area variance.  When ``A`` comes out negative -
    the raw rates vary *less* than Poisson noise alone would make them - it is
    set to zero, so every area takes the overall rate.  esda's
    ``Empirical_Bayes`` leaves a negative ``A`` in place, which produces
    negative shrinkage weights; the clamp is the convention GeoDa and Marshall
    use, and the report says when it fired.
    """
    e, b = _validate_rate_inputs(events, base)
    rate = e / b
    m = e.sum() / b.sum()
    s2 = float(np.sum(b * (rate - m) ** 2) / b.sum())
    a_raw = s2 - m / b.mean()
    a = max(a_raw, 0.0)
    weight = a / (a + m / b) if a > 0 else np.zeros_like(b)
    smoothed = weight * rate + (1.0 - weight) * m
    return {
        "rate": smoothed,
        "shrink_weight": weight,
        "overall_rate": float(m),
        "between_variance": float(a_raw),
        "variance_clamped": bool(a_raw < 0),
    }


def spatial_rate(events, base, graph: SpatialGraph) -> np.ndarray:
    """Window rate: events and bases of the unit plus its neighbours."""
    e, b = _validate_rate_inputs(events, base)
    gb = graph.binary()
    return (e + gb.lag(e)) / (b + gb.lag(b))


def spatial_eb_rate(events, base, graph: SpatialGraph) -> dict:
    """Spatial (local) empirical Bayes rate - Marshall's estimator with the
    prior taken from each unit's own neighbourhood (unit plus neighbours)
    instead of the whole study area.  Same formula as esda's
    ``Spatial_Empirical_Bayes``: a negative local variance is set to zero.
    """
    e, b = _validate_rate_inputs(events, base)
    rate = e / b
    prior = spatial_rate(e, b, graph)
    gb = graph.binary()
    window_b = gb.lag(b) + b
    count = graph.cardinality + 1.0
    # sum over window of b_j (r_j - prior_i)^2, expanded so it stays a lag
    sum_br2 = gb.lag(b * rate ** 2) + b * rate ** 2
    sum_br = gb.lag(b * rate) + b * rate
    var_left = (sum_br2 - 2.0 * prior * sum_br + prior ** 2 * window_b) / window_b
    var_right = prior / (window_b / count)
    var = np.maximum(var_left - var_right, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = np.where(var > 0, var / (var + prior / b), 0.0)
    return {"rate": prior + (rate - prior) * weight, "shrink_weight": weight, "prior": prior}


def assuncao_reis_rate(events, base) -> np.ndarray:
    """Assunção & Reis (1999) EB-standardised rate, a z-like score.

    ``z_i = (r_i - b) / sqrt(a + b / P_i)`` with ``b`` the overall rate and
    ``a`` the method-of-moments variance; where ``a + b / P_i`` is negative
    the variance falls back to ``b / P_i``.  This is the variable GeoDa and
    esda (``assuncao_rate``) feed to Moran's I for rates: it removes the
    variance instability small denominators put into raw rates.
    """
    e, b = _validate_rate_inputs(events, base)
    y = e / b
    overall = e.sum() / b.sum()
    s2 = float(np.sum(b * (y - overall) ** 2) / b.sum())
    a = s2 - overall / (b.sum() / len(e))
    v_raw = a + overall / b
    v = np.where(v_raw < 0, overall / b, v_raw)
    return (y - overall) / np.sqrt(v)


def rate_smoothing_summary(events, base, graph: SpatialGraph | None) -> dict:
    e, b = _validate_rate_inputs(events, base)
    raw = e / b
    geb = global_eb_rate(e, b)
    out = {
        "raw": raw,
        "global_eb": geb["rate"],
        "global_weight": geb["shrink_weight"],
        "overall_rate": geb["overall_rate"],
        "between_variance": geb["between_variance"],
        "variance_clamped": geb["variance_clamped"],
        "assuncao_reis": assuncao_reis_rate(e, b),
    }
    if graph is not None:
        out["spatial"] = spatial_rate(e, b, graph)
        seb = spatial_eb_rate(e, b, graph)
        out["spatial_eb"] = seb["rate"]
        out["spatial_weight"] = seb["shrink_weight"]
    # Variance stability: how strongly raw-rate extremes sit on small bases.
    order = np.argsort(b)
    small = order[: max(1, len(b) // 5)]
    large = order[-max(1, len(b) // 5):]
    out["raw_sd_small_base"] = float(np.std(raw[small]))
    out["raw_sd_large_base"] = float(np.std(raw[large]))
    return out


# ---------------------------------------------------------------------------
# Moran (global and local) on an arbitrary variable, permutation inference
# ---------------------------------------------------------------------------


def moran_global_perm(y, graph: SpatialGraph, permutations: int = 999, seed: int = 42) -> dict:
    """Global Moran's I with row-standardised weights and total randomisation."""
    y = np.asarray(y, dtype=float)
    wr = graph.row_standardized()
    z = y - y.mean()
    den = float(z @ z)
    s0 = float(wr.weight.sum())
    if den == 0 or s0 == 0:
        return {"I": float("nan"), "p": float("nan"), "z": float("nan"), "expected": -1.0 / (len(y) - 1)}

    def stat(v):
        return len(v) / s0 * float(v @ wr.lag(v)) / den

    observed = stat(z)
    rng = np.random.default_rng(int(seed))
    sims = np.array([stat(rng.permutation(z)) for _ in range(int(permutations))])
    sd = sims.std()
    return {
        "I": float(observed),
        "expected": -1.0 / (len(y) - 1),
        "sim_mean": float(sims.mean()) if sims.size else float("nan"),
        "z": float((observed - sims.mean()) / sd) if sd > 0 else 0.0,
        "p": two_sided_p(observed, sims),
    }


def moran_local_perm(y, graph: SpatialGraph, permutations: int = 999, seed: int = 42) -> dict:
    """Local Moran's I, ``I_i = (n-1) z_i lag_i / sum z^2``, as in esda.

    Returns the statistic, two-sided conditional-permutation p-values, the
    z-score against the permutation distribution and the HH/LH/LL/HL quadrant
    (assigned for every unit; callers mask non-significant ones).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    wr = graph.row_standardized()
    z = y - y.mean()
    den = float(z @ z)
    lag = wr.lag(z)
    scale = (n - 1) / den if den > 0 else 0.0
    local_i = scale * z * lag
    p = np.full(n, np.nan)
    zsim = np.full(n, np.nan)
    for i, draws, w in conditional_draws(wr, permutations, seed):
        sims = scale * z[i] * (z[draws] @ w)
        p[i] = two_sided_p(local_i[i], sims)
        sd = sims.std()
        zsim[i] = (local_i[i] - sims.mean()) / sd if sd > 0 else 0.0
    return {"I": local_i, "p": p, "z": zsim, "lag": lag, "quadrant": quadrant_labels(z, lag)}


# ---------------------------------------------------------------------------
# Multivariate local Geary (Anselin 2019)
# ---------------------------------------------------------------------------


def multivariate_local_geary(data, graph: SpatialGraph, permutations: int = 999, seed: int = 42) -> dict:
    """``c_i = (1/k) sum_v sum_j w_ij (z_vi - z_vj)^2`` on z-scored variables.

    Row-standardised weights, variables z-scored with the population standard
    deviation, exactly as esda's ``Geary_Local_MV``.  A significant ``c_i``
    *below* its permutation mean is positive local association (the unit
    resembles its neighbours in attribute space); *above* it is negative
    association (a multivariate local outlier).
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("Multivariate local Geary needs at least two variables.")
    n, k = x.shape
    z = np.column_stack([_zscores(x[:, v]) for v in range(k)])
    wr = graph.row_standardized()
    diff2 = np.sum((z[wr.focal] - z[wr.neighbor]) ** 2, axis=1)
    c = np.bincount(wr.focal, weights=wr.weight * diff2, minlength=n) / k
    p = np.full(n, np.nan)
    sim_mean = np.full(n, np.nan)
    for i, draws, w in conditional_draws(wr, permutations, seed):
        d = np.sum((z[draws] - z[i]) ** 2, axis=2)  # (P, k_i)
        sims = (d @ w) / k
        p[i] = two_sided_p(c[i], sims)
        sim_mean[i] = sims.mean()
    association = np.where(c < sim_mean, "Positive", np.where(c > sim_mean, "Negative", "Not Significant"))
    return {"c": c, "p": p, "sim_mean": sim_mean, "association": association.tolist()}


# ---------------------------------------------------------------------------
# Local join counts (Anselin & Li 2019)
# ---------------------------------------------------------------------------


def _binary(values, name):
    v = np.asarray(values, dtype=float)
    bad = ~np.isin(v, (0.0, 1.0))
    if np.any(bad):
        raise ValueError(f"{name} must contain only 0 and 1 (found {np.unique(v[bad])[:5].tolist()}).")
    return v


def local_join_counts(x, graph: SpatialGraph, permutations: int = 999, seed: int = 42,
                      z=None, mode: str = "univariate", extra=None) -> dict:
    """Local join counts with binary weights and an upper-tail permutation test.

    ``univariate``  ``LJC_i = x_i * sum_j w_ij x_j``
    ``bjc``         bivariate join: focal has x=1, z=0; counts neighbours with
                    x=0, z=1 (Anselin & Li 2019, case 1)
    ``clc``         co-location: focal has x=1, z=1; counts neighbours with both
    ``multivariate`` the co-location of every listed variable

    The statistic is undefined (NaN p) where the focal unit cannot form the
    join at all or forms none.  Unlike esda, whose case-1 simulation omits
    the ``x_j = 0`` condition, the simulated statistic here uses exactly the
    observed definition, with ``(x_j, z_j)`` pairs permuted together.
    """
    xb = _binary(x, "The first variable")
    gb = graph.binary()
    n = len(xb)
    if mode == "univariate":
        focal_ok = xb == 1
        neigh_flag = xb
    elif mode in ("bjc", "clc"):
        if z is None:
            raise ValueError("A second binary variable is required for bivariate join counts.")
        zb = _binary(z, "The second variable")
        if mode == "bjc":
            focal_ok = (xb == 1) & (zb == 0)
            neigh_flag = ((xb == 0) & (zb == 1)).astype(float)
        else:
            focal_ok = (xb == 1) & (zb == 1)
            neigh_flag = ((xb == 1) & (zb == 1)).astype(float)
    elif mode == "multivariate":
        stack = [xb] + [_binary(v, f"Variable {k + 2}") for k, v in enumerate(extra or [])]
        if len(stack) < 2:
            raise ValueError("Multivariate join counts need at least two binary variables.")
        prod = np.prod(np.vstack(stack), axis=0)
        focal_ok = prod == 1
        neigh_flag = prod
    else:
        raise ValueError(f"Unknown join-count mode: {mode}")
    ljc = np.where(focal_ok, gb.lag(neigh_flag), 0.0)
    p = np.full(n, np.nan)
    for i, draws, w in conditional_draws(gb, permutations, seed):
        if not focal_ok[i] or ljc[i] == 0:
            continue
        sims = neigh_flag[draws] @ w
        p[i] = greater_p(ljc[i], sims)
    return {"ljc": ljc, "p": p, "eligible": focal_ok}


# ---------------------------------------------------------------------------
# LOSH (Ord & Getis 2012)
# ---------------------------------------------------------------------------


def losh(y, graph: SpatialGraph, a: float = 2.0, include_self: bool = True) -> dict:
    """Local spatial heteroskedasticity ``H_i`` with chi-square inference.

    Residuals are taken about each unit's own local mean,
    ``e_j = x_j - xbar_j`` - the form both spdep's ``LOSH`` and esda's ``LOSH``
    implement; Ord & Getis (2012) print ``e_j = x_j - xbar_i`` about the focal
    unit's mean, which the manual card discusses.  ``include_self`` adds the
    focal unit to its own neighbourhood with weight 1 (the Gi*-style window
    Ord & Getis use).  The chi-square approximation (``Z_i = 2H_i/V_i`` on
    ``2/V_i`` degrees of freedom) is derived for ``a = 2``; for any other ``a``
    p-values are returned as NaN rather than as numbers that look valid.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    g = graph.binary()
    if include_self:
        g = g.with_self(1.0)
    rowsum = g.rowsum
    safe = np.where(rowsum > 0, rowsum, 1.0)
    ylag = g.lag(y) / safe
    resid = np.abs(y - ylag) ** a
    h1 = float(resid.mean())
    if h1 == 0:
        raise ValueError("Every local residual is zero - the field is locally constant.")
    denom = h1 * safe
    hi = g.lag(resid) / denom
    sq_rowsum = np.bincount(g.focal, weights=g.weight ** 2, minlength=n)
    h2 = float(np.mean(resid ** 2))
    var = (1.0 / (n - 1)) * denom ** -2 * (h2 - h1 ** 2) * (n * sq_rowsum - rowsum ** 2)
    p = np.full(n, np.nan)
    if abs(a - 2.0) < 1e-12:
        for i in range(n):
            if rowsum[i] > 0 and var[i] > 0:
                dof = 2.0 / var[i]
                p[i] = chi2_sf(2.0 * hi[i] / var[i], dof)
    return {"H": hi, "local_mean": ylag, "resid": resid, "var": var, "p": p}


def getis_ord_star(y, graph: SpatialGraph) -> dict:
    """Gi* z-score with binary weights including the unit itself (Ord & Getis 1995).

    ``G*_i = (sum_j w_ij x_j - xbar W_i) / (S sqrt((n S1_i - W_i^2) / (n - 1)))``
    The z-score is the statistic itself; p-values are two-sided normal.
    """
    x = np.asarray(y, dtype=float)
    n = len(x)
    g = graph.binary().with_self(1.0)
    w_i = g.rowsum
    s1 = np.bincount(g.focal, weights=g.weight ** 2, minlength=n)
    xbar = x.mean()
    s = math.sqrt(float(np.mean(x ** 2) - xbar ** 2))
    den = s * np.sqrt(np.maximum(n * s1 - w_i ** 2, 0.0) / (n - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(den > 0, (g.lag(x) - xbar * w_i) / den, 0.0)
    p = np.array([2.0 * normal_sf(abs(v)) for v in z])
    return {"z": z, "p": p}


# ---------------------------------------------------------------------------
# Lee's L / spatial Pearson (Lee 2001)
# ---------------------------------------------------------------------------


def lee_l(x, y, graph: SpatialGraph, permutations: int = 999, seed: int = 42,
          include_self: bool = True) -> dict:
    """Lee's (2001) bivariate spatial association, global and local.

    Both variables are z-scored and spatially smoothed with the row-
    standardised matrix V (``x~ = V z_x``), and

        L   = sum_i x~_i y~_i / sum_i (sum_j v_ij)^2      (global)
        L_i = x~_i y~_i                                    (local, V row-standardised)

    which is esda's ``Spatial_Pearson`` / ``Spatial_Pearson_Local``.  Lee
    recommends a smoother that includes the unit itself (``include_self``).
    Because both variables are smoothed, L is symmetric in x and y - the
    property that separates it from bivariate Moran's I, which smooths only
    one of them.

    Inference: the global statistic permutes the (x, y) pairs over locations
    (``p``) - the test of a spatial co-pattern beyond the aspatial
    correlation, as in esda - and, separately, shuffles y alone against a
    fixed x (``p_any``), the test of any bivariate spatial association.  Each
    local statistic holds unit i fixed and redraws its neighbours.
    """
    zx, zy = _zscores(x), _zscores(y)
    n = len(zx)
    # Lee's smoother: binary connectivity, the unit itself optionally added
    # with the same weight as a neighbour, then row-standardised.  Adding a
    # self-weight of 1 to weights that are already row-standardised would give
    # the unit half of its own window.
    g = graph.binary()
    if include_self:
        g = g.with_self(1.0)
    v = g.row_standardized()
    sx, sy = v.lag(zx), v.lag(zy)
    denom = float(np.sum(v.rowsum ** 2))
    if denom == 0:
        raise ValueError("No unit has a neighbour; Lee's L is undefined.")

    def glob(a, b):
        return float(np.dot(v.lag(a), v.lag(b)) / denom)

    observed = float(np.dot(sx, sy) / denom)
    lxx = float(np.dot(sx, sx) / denom)
    lyy = float(np.dot(sy, sy) / denom)
    rng = np.random.default_rng(int(seed))
    sims = []
    for _ in range(int(permutations)):
        order = rng.permutation(n)
        sims.append(glob(zx[order], zy[order]))
    sims = np.asarray(sims)
    sd = sims.std() if sims.size else 0.0
    sims_any = np.array([glob(zx, zy[rng.permutation(n)]) for _ in range(int(permutations))])
    local = sx * sy
    p_local = np.full(n, np.nan)
    # Unit i stays inside its own window; only its other neighbours are
    # redrawn.  The draws come from the graph without self-links, whose
    # cardinality is exactly the number of non-self weights in V's row.
    for i, draws, _w in conditional_draws(graph, permutations, seed + 1):
        nbr, wts = v.neighbors_of(i)
        self_mask = nbr == i
        w_self = float(wts[self_mask].sum()) if np.any(self_mask) else 0.0
        w_other = wts[~self_mask]
        k = min(len(w_other), draws.shape[1])
        d = draws[:, :k]
        sim_x = w_self * zx[i] + zx[d] @ w_other[:k]
        sim_y = w_self * zy[i] + zy[d] @ w_other[:k]
        p_local[i] = two_sided_p(local[i], sim_x * sim_y)
    r_raw = float(np.corrcoef(zx, zy)[0, 1]) if n > 2 else float("nan")
    r_smooth = float(np.corrcoef(sx, sy)[0, 1]) if np.std(sx) > 0 and np.std(sy) > 0 else float("nan")
    return {
        "L": observed,
        "L_xx": lxx,
        "L_yy": lyy,
        "pearson_r": r_raw,
        "smoothed_r": r_smooth,
        "sim_mean": float(sims.mean()) if sims.size else float("nan"),
        "z": float((observed - sims.mean()) / sd) if sd > 0 else 0.0,
        "p": two_sided_p(observed, sims),
        "p_any": two_sided_p(observed, sims_any),
        "sim_sd": float(sd),
        "local": local,
        "local_p": p_local,
        "smoothed_x": sx,
        "smoothed_y": sy,
        "local_class": quadrant_labels(sx, sy),
    }


# ---------------------------------------------------------------------------
# Partial and conditional local Moran (Wolf 2024, esda.moran_local_mv)
# ---------------------------------------------------------------------------


def _unit_scale(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0)
    sd = a.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return a / sd


def moran_local_partial(y, controls, graph: SpatialGraph, permutations: int = 999,
                        seed: int = 42) -> dict:
    """Local partial Moran - y's local association with its lag, net of controls.

    With ``D = [1, z, X]`` and ``R = Wz``, the regression of the lag on D has
    a slope for z equal to ``sum_i [(D'D)^-1 D_i']_z R_i``; each term (times
    n-1) is that unit's share of the partial slope.  Matches esda's
    ``MoranLocalPartial`` with unit scaling and partial labels.
    """
    z = _unit_scale(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()
    X = _unit_scale(np.asarray(controls, dtype=float).reshape(len(z), -1))
    n = len(z)
    wr = graph.row_standardized()
    lag = wr.lag(z)
    D = np.column_stack([np.ones(n), z, X])
    DtDi = np.linalg.inv(D.T @ D)
    left = (D @ DtDi)[:, 1] * (n - 1)
    association = left * lag
    p = np.full(n, np.nan)
    for i, draws, w in conditional_draws(wr, permutations, seed):
        sims = left[i] * (z[draws] @ w)
        p[i] = two_sided_p(association[i], sims)
    labels = quadrant_labels(left - left.mean(), lag)
    return {"association": association, "p": p, "left": left, "lag": lag, "label": labels}


def moran_local_conditional(y, controls, graph: SpatialGraph, permutations: int = 999,
                            seed: int = 42) -> dict:
    """Local conditional Moran - local Moran of y after regressing out controls.

    ``y_f`` is the OLS residual of the unit-scaled y on the unit-scaled
    controls (with intercept); ``I_i = (n-1) y_f,i (W y_f)_i / y_f' y_f``.
    Matches esda's ``MoranLocalConditional``.
    """
    z = _unit_scale(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()
    X = _unit_scale(np.asarray(controls, dtype=float).reshape(len(z), -1))
    n = len(z)
    D = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(D, z, rcond=None)
    yf = z - D @ beta
    wr = graph.row_standardized()
    lag = wr.lag(yf)
    den = float(yf @ yf)
    scale = (n - 1) / den if den > 0 else 0.0
    association = scale * yf * lag
    p = np.full(n, np.nan)
    for i, draws, w in conditional_draws(wr, permutations, seed):
        sims = scale * yf[i] * (yf[draws] @ w)
        p[i] = two_sided_p(association[i], sims)
    r2 = 1.0 - den / float(z @ z) if float(z @ z) > 0 else float("nan")
    return {"association": association, "p": p, "filtered": yf, "lag": lag,
            "label": quadrant_labels(yf, lag), "controls_r2": r2}


# ---------------------------------------------------------------------------
# Gamma index (Hubert, Golledge & Costanzo 1981)
# ---------------------------------------------------------------------------


GAMMA_OPERATIONS = {
    "c": "cross-product  sum w_ij y_i y_j",
    "s": "squared difference  sum w_ij (y_i - y_j)^2",
    "a": "absolute difference  sum w_ij |y_i - y_j|",
}


def gamma_index(y, graph: SpatialGraph, operation: str = "c", standardize: bool = False,
                permutations: int = 999, seed: int = 42) -> dict:
    """Generalised cross-product statistic ``Gamma = sum_ij w_ij a_ij``.

    ``a_ij`` is the attribute similarity named by ``operation``; weights are
    used as given (binary contiguity is the textbook case).  Inference is
    total randomisation.  For the cross-product, a large Gamma is positive
    autocorrelation; for the two difference measures, a *small* Gamma is -
    the report states the direction rather than leaving it to the reader.
    """
    if operation not in GAMMA_OPERATIONS:
        raise ValueError(f"Unknown Gamma operation: {operation}")
    v = np.asarray(y, dtype=float)
    if standardize:
        v = _zscores(v)
    f, nb, w = graph.focal, graph.neighbor, graph.weight

    def stat(vals):
        if operation == "c":
            return float(np.sum(w * vals[f] * vals[nb]))
        if operation == "s":
            return float(np.sum(w * (vals[f] - vals[nb]) ** 2))
        return float(np.sum(w * np.abs(vals[f] - vals[nb])))

    observed = stat(v)
    rng = np.random.default_rng(int(seed))
    sims = np.array([stat(rng.permutation(v)) for _ in range(int(permutations))])
    sd = sims.std() if sims.size else 0.0
    return {
        "gamma": observed,
        "sim_mean": float(sims.mean()) if sims.size else float("nan"),
        "sim_min": float(sims.min()) if sims.size else float("nan"),
        "sim_max": float(sims.max()) if sims.size else float("nan"),
        "z": float((observed - sims.mean()) / sd) if sd > 0 else 0.0,
        "p": two_sided_p(observed, sims),
        "positive_when": "large" if operation == "c" else "small",
        "simulated": sims,
    }


# ---------------------------------------------------------------------------
# S-maup (Duque, Laniado & Polo 2018)
# ---------------------------------------------------------------------------

_SMAUP_N = np.array([25, 100, 225, 400, 625, 900], dtype=float)
_SMAUP_RHO = np.array([-0.9, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.9])
# Critical values, Duque et al. (2018) Table 3, rows by rho, columns by n.
_SMAUP_CV = {
    0.01: np.array([
        [0.83702, 0.09218, 0.23808, 0.05488, 0.07218, 0.02621],
        [0.83676, 0.16134, 0.13402, 0.06737, 0.05486, 0.02858],
        [0.83597, 0.16524, 0.13446, 0.06616, 0.06247, 0.02851],
        [0.83316, 0.19276, 0.13396, 0.06330, 0.06090, 0.03696],
        [0.82370, 0.17925, 0.15514, 0.07732, 0.07988, 0.09301],
        [0.76472, 0.23404, 0.24640, 0.11588, 0.10715, 0.07070],
        [0.67337, 0.28921, 0.25535, 0.13992, 0.12975, 0.09856],
        [0.52155, 0.47399, 0.29351, 0.23923, 0.20321, 0.16250],
        [0.28599, 0.28938, 0.43520, 0.44060, 0.34437, 0.55967],
    ]),
    0.05: np.array([
        [0.83699, 0.08023, 0.10962, 0.04894, 0.04641, 0.02423],
        [0.83662, 0.12492, 0.08643, 0.05900, 0.04280, 0.02459],
        [0.83578, 0.13796, 0.08679, 0.05927, 0.04260, 0.02658],
        [0.78849, 0.16932, 0.08775, 0.05464, 0.04787, 0.03042],
        [0.81952, 0.15746, 0.11126, 0.06961, 0.06066, 0.05234],
        [0.70466, 0.21088, 0.15360, 0.09766, 0.07938, 0.06461],
        [0.59461, 0.23497, 0.18244, 0.11682, 0.10129, 0.08860],
        [0.48958, 0.37226, 0.22280, 0.20540, 0.16144, 0.14123],
        [0.21580, 0.22532, 0.27122, 0.29043, 0.23648, 0.31424],
    ]),
    0.10: np.array([
        [0.69331, 0.06545, 0.07858, 0.04015, 0.03374, 0.02187],
        [0.79421, 0.09566, 0.06777, 0.05058, 0.03392, 0.02272],
        [0.68900, 0.10707, 0.07039, 0.05151, 0.03609, 0.02411],
        [0.73592, 0.14282, 0.07076, 0.04649, 0.04001, 0.02614],
        [0.71632, 0.13621, 0.08801, 0.06112, 0.04937, 0.03759],
        [0.63718, 0.18239, 0.12101, 0.08324, 0.06347, 0.05549],
        [0.46548, 0.17541, 0.14248, 0.10008, 0.08137, 0.07701],
        [0.34720, 0.28774, 0.18170, 0.16442, 0.13395, 0.12354],
        [0.17640, 0.18835, 0.21695, 0.23031, 0.19435, 0.22411],
    ]),
}


def smaup_statistic(n: int, k: int, rho: float) -> dict:
    """S-maup statistic and its tabulated critical values.

    ``S = L / (1 + eta * exp(rho * tau))`` with ``theta = k/n``,
    ``L = 1/(1 + exp(-2.2 + 7.03 theta))``, ``eta = exp(-0.6618) theta^1.3``
    and ``tau = 5.32 - 5.53 theta`` (Duque et al. 2018, eq. 10-13).  The
    critical value comes from the table row whose rho is *nearest* to the
    observed one, interpolated linearly in n.  esda picks the row through
    open-interval tests, so a rho of exactly 0.8, 0.6, 0.4 or 0.15 - or any
    rho of 1.0 - falls through to the -0.9 row; nearest-row selection has no
    such gap.  n outside 25-900 is clamped and flagged, as in the paper.
    """
    if n < 2 or k < 1 or k > n:
        raise ValueError("S-maup needs 1 <= k <= n and at least two areas.")
    warnings = []
    n_eff = float(n)
    if n_eff < 25:
        warnings.append("n is below 25, the smallest size in the simulation study; the result is extrapolated.")
        n_eff = 25.0
    elif n_eff > 900:
        warnings.append("n is above 900, the largest size in the simulation study; the result is extrapolated.")
        n_eff = 900.0
    theta = float(k) / float(n_eff)
    big_l = 1.0 / (1.0 + math.exp(-2.2 + theta * 7.03))
    eta = math.exp(-0.6618) * theta ** 1.3
    tau = theta * -5.53 + 5.32
    s = big_l / (1.0 + eta * math.exp(rho * tau))
    row = int(np.argmin(np.abs(_SMAUP_RHO - float(rho))))
    crit = {alpha: float(np.interp(n_eff, _SMAUP_N, table[row])) for alpha, table in _SMAUP_CV.items()}
    if s > crit[0.01]:
        verdict, level = "MAUP effect significant at the 1% level", 0.01
    elif s > crit[0.05]:
        verdict, level = "MAUP effect significant at the 5% level", 0.05
    elif s > crit[0.10]:
        verdict, level = "MAUP effect significant at the 10% level", 0.10
    else:
        verdict, level = "No significant MAUP effect (p > 0.10)", None
    return {"smaup": float(s), "theta": theta, "critical": crit, "rho_row": float(_SMAUP_RHO[row]),
            "verdict": verdict, "level": level, "warnings": warnings}
