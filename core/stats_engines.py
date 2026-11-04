# -*- coding: utf-8 -*-
"""Spatial statistics engines for calculations."""
from __future__ import annotations

import logging
import math
import importlib.util
from typing import Optional

import numpy as np

logger = logging.getLogger("PlanX GeoStats Lab")

# Try importing PySAL modules
HAS_PYQ = all(importlib.util.find_spec(name) is not None for name in ("esda", "libpysal"))


class InputError(ValueError):
    """The input cannot support the statistic; the text is a sentence for the user.

    Raised for weights without a single neighbour pair (S0 = 0), weights in
    which every feature is an island, and an analysis field without variation.
    Algorithms turn it into a ``QgsProcessingException`` with the same text.
    """


MSG_NO_NEIGHBOURS = (
    "No feature has a neighbour under the chosen spatial weights, so {stat} cannot be computed. "
    "Use K nearest neighbours, or a distance band that is larger than the gap between features."
)
MSG_NO_VARIATION = (
    "Every value of the analysis field is the same, so {stat} has no variation to test. "
    "Choose a field whose values differ between features."
)


def _no_neighbours(stat: str) -> InputError:
    return InputError(MSG_NO_NEIGHBOURS.format(stat=stat))


def _no_variation(stat: str) -> InputError:
    return InputError(MSG_NO_VARIATION.format(stat=stat))


def _too_few(stat: str, minimum: int, n: int) -> InputError:
    return InputError(
        f"{stat} needs at least {minimum} features with a value; only {n} were found."
    )


def weights_to_arrays(
    neighbors: dict,
    weights: Optional[dict],
    id_order: list,
    binary: bool = False,
    drop_self: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbour dictionaries as sparse (row, col, weight) arrays over ``id_order``.

    Neighbours outside ``id_order`` are skipped, a repeated pair keeps its last
    weight (as a dense matrix assignment would) and, with ``drop_self``, the
    diagonal is removed. ``binary`` sets every kept weight to 1.
    """
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    weights = weights or {}
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, []) or []
        w_list = weights.get(fid, []) or []
        for k, j_fid in enumerate(neighs):
            j = id_to_idx.get(j_fid)
            if j is None or (drop_self and j == i):
                continue
            rows.append(i)
            cols.append(j)
            vals.append(1.0 if binary or k >= len(w_list) else float(w_list[k]))
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    v = np.asarray(vals, dtype=float)
    if r.size:
        n = len(id_order)
        keys = r * n + c
        # Last occurrence of each pair wins: unique() on the reversed keys.
        _, first_rev = np.unique(keys[::-1], return_index=True)
        keep = np.sort(r.size - 1 - first_rev)
        r, c, v = r[keep], c[keep], v[keep]
    return r, c, v


def _require_finite(x, stat: str) -> None:
    if not np.all(np.isfinite(np.asarray(x, dtype=float))):
        raise InputError(
            f"The analysis field contains missing or non-numeric values; remove them before running {stat}."
        )


def _require_neighbours(neighbors, weights, id_order, stat: str) -> None:
    """InputError unless at least one pair of distinct features has a non-zero weight."""
    _, _, vals = weights_to_arrays(neighbors, weights, id_order)
    if vals.size == 0 or not np.any(vals != 0):
        raise _no_neighbours(stat)


def _normal_two_sided(z):
    """Two-sided standard normal p-value, element-wise; NaN stays NaN."""
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z)
    out[ok] = [math.erfc(abs(v) / math.sqrt(2.0)) for v in z[ok]]
    return out


def _sparse_weight_sums(rows, cols, vals, n: int):
    """S0, A1 = sum w_ij^2, A2 = sum w_ij w_ji, row sums and column sums of sparse weights.

    Pairs must be unique (``weights_to_arrays`` guarantees it). A2 matches each
    (i, j) with its transpose through sorted integer keys: O(nnz log nnz).
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=float)
    s0 = float(np.sum(vals))
    a1 = float(np.sum(vals ** 2))
    a2 = 0.0
    if vals.size:
        keys = rows * n + cols
        order = np.argsort(keys, kind="mergesort")
        sorted_keys = keys[order]
        rev = cols * n + rows
        pos = np.minimum(np.searchsorted(sorted_keys, rev), sorted_keys.size - 1)
        hit = sorted_keys[pos] == rev
        a2 = float(np.sum(vals[hit] * vals[order[pos[hit]]]))
    row_sums = np.bincount(rows, weights=vals, minlength=n)
    col_sums = np.bincount(cols, weights=vals, minlength=n)
    return s0, a1, a2, row_sums, col_sums


def calculate_getis_ord(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    star: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Getis-Ord local statistic as a z-score, with binary weights (Ord & Getis 1995).

    ``star=True`` (Gi*) counts the feature itself:
        G*_i = (sum_j w_ij x_j - xbar W_i) / (S sqrt((n S1_i - W_i^2) / (n - 1)))
    ``star=False`` (Gi) leaves feature i out of the sums, the mean and the
    standard deviation, and uses n - 1 features:
        G_i = (sum_{j!=i} w_ij x_j - xbar(i) W_i) / (S(i) sqrt(((n-1) S1_i - W_i^2) / (n - 2)))
    Supplied weights only decide who is a neighbour; the statistic is binary,
    so row standardisation cannot change it.

    A feature with no neighbour besides itself gets NaN z and p and bin 0.
    Raises InputError when every feature is an island or the field is constant.
    """
    x = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        raise _too_few("Getis-Ord Gi*" if star else "Getis-Ord Gi", 3, n)
    _require_finite(x, "Getis-Ord Gi*" if star else "Getis-Ord Gi")
    if float(np.std(x)) == 0.0:
        raise _no_variation("Getis-Ord Gi*" if star else "Getis-Ord Gi")

    rows, cols, _ = weights_to_arrays(neighbors, weights, id_order, binary=True)
    w_i = np.bincount(rows, minlength=n).astype(float)  # neighbours other than i
    if not np.any(w_i > 0):
        raise _no_neighbours("Getis-Ord Gi*" if star else "Getis-Ord Gi")
    lag = np.bincount(rows, weights=x[cols], minlength=n)

    with np.errstate(invalid="ignore", divide="ignore"):
        if star:
            w_star = w_i + 1.0
            xbar = float(np.mean(x))
            s = math.sqrt(max(float(np.mean(x ** 2)) - xbar ** 2, 0.0))
            den = s * np.sqrt((n * w_star - w_star ** 2) / (n - 1))
            num = lag + x - xbar * w_star
        else:
            sx, sx2 = float(np.sum(x)), float(np.sum(x ** 2))
            xbar_i = (sx - x) / (n - 1)
            s_i = np.sqrt(np.maximum((sx2 - x ** 2) / (n - 1) - xbar_i ** 2, 0.0))
            den = s_i * np.sqrt(((n - 1) * w_i - w_i ** 2) / (n - 2))
            num = lag - xbar_i * w_i
        z_scores = np.where((w_i > 0) & (den > 0), num / den, np.nan)

    p_values = _normal_two_sided(z_scores)
    conf_bins = np.zeros(n, dtype=int)
    ok = np.isfinite(z_scores)
    sign = np.where(z_scores > 0, 1, -1)
    for level, cut in ((3, 0.01), (2, 0.05), (1, 0.10)):
        hit = ok & (conf_bins == 0) & (p_values < cut)
        conf_bins[hit] = level * sign[hit]
    return z_scores, p_values, conf_bins


def calculate_mean_center(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> tuple[float, float]:
    """Calculates the mean center of coordinate pairs."""
    if weights is None or len(weights) == 0:
        return float(np.mean(x_coords)), float(np.mean(y_coords))

    total_weight = np.sum(weights)
    if total_weight == 0:
        return float(np.mean(x_coords)), float(np.mean(y_coords))

    mean_x = np.sum(x_coords * weights) / total_weight
    mean_y = np.sum(y_coords * weights) / total_weight
    return float(mean_x), float(mean_y)


def calculate_central_feature(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> int:
    """Finds the index of the central feature based on minimum total distance."""
    n = len(x_coords)
    if n <= 1:
        return 0

    coords = np.column_stack((x_coords, y_coords))
    # Pairwise Euclidean distances
    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    if weights is None or len(weights) == 0:
        dist_sums = dists.sum(axis=1)
    else:
        # Weighted distance sum
        dist_sums = (dists * weights[None, :]).sum(axis=1)

    return int(np.argmin(dist_sums))


def calculate_sde(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
    num_std: int = 1
) -> tuple[float, float, float, float, float]:
    """Calculates Standard Deviational Ellipse (SDE) parameters.

    Returns:
        A tuple of (mean_x, mean_y, rotation_angle_radians, semi_major_axis, semi_minor_axis)
    """
    n = len(x_coords)
    mean_x, mean_y = calculate_mean_center(x_coords, y_coords, weights)

    if n <= 2:
        return mean_x, mean_y, 0.0, 0.0, 0.0

    x_prime = x_coords - mean_x
    y_prime = y_coords - mean_y

    W = np.ones(n) if (weights is None or len(weights) == 0) else weights
    sum_w = np.sum(W)
    if sum_w == 0:
        W = np.ones(n)
        sum_w = n

    sum_x2 = np.sum(W * (x_prime ** 2))
    sum_y2 = np.sum(W * (y_prime ** 2))
    sum_xy = np.sum(W * x_prime * y_prime)

    # Calculate rotation angle theta
    # Using the standardPrincipal Orientation formula
    theta = 0.5 * np.arctan2(2 * sum_xy, sum_x2 - sum_y2)

    # Standard deviations along rotated axes
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    std_x = np.sqrt(np.sum(W * (x_prime * cos_t - y_prime * sin_t) ** 2) / sum_w)
    std_y = np.sqrt(np.sum(W * (x_prime * sin_t + y_prime * cos_t) ** 2) / sum_w)

    # Semi-major/minor axes scaling
    semi_x = num_std * std_x
    semi_y = num_std * std_y

    # Let semi_major be the larger one
    if semi_x >= semi_y:
        semi_major = semi_x
        semi_minor = semi_y
        angle = theta
    else:
        semi_major = semi_y
        semi_minor = semi_x
        angle = theta + np.pi / 2.0  # Align rotation to semi-major axis

    return mean_x, mean_y, float(angle), float(semi_major), float(semi_minor)


def calculate_local_moran(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Calculates Anselin Local Moran's I cluster and outlier diagnostics.

    Returns:
        A tuple of:
          - I_values: NumPy array of Moran's I indices (floats)
          - z_scores: NumPy array of z-scores (floats)
          - p_values: NumPy array of p-values (floats)
          - quadrants: List of strings ('HH', 'LL', 'HL', 'LH', 'Not Significant')
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    I_values = np.zeros(n)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    quadrants = ["Not Significant"] * n

    if n <= 2:
        raise _too_few("Local Moran's I", 3, n)
    _require_finite(y, "Local Moran's I")

    y_mean = np.mean(y)
    z = y - y_mean
    m2 = np.sum(z ** 2) / n

    if m2 == 0:
        raise _no_variation("Local Moran's I")
    _require_neighbours(neighbors, weights, id_order, "Local Moran's I")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    b2 = (n * np.sum(z ** 4)) / (np.sum(z ** 2) ** 2)  # Kurtosis

    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])

        valid_neigh_indices = []
        valid_w = []
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx:
                valid_neigh_indices.append(id_to_idx[nid])
                valid_w.append(f_weights[j])

        w_sum = sum(valid_w)
        w_sum2 = sum(w**2 for w in valid_w)

        if w_sum == 0:
            continue

        # Spatial lag
        spatial_lag = np.sum(np.array(valid_w) * z[valid_neigh_indices])
        I_i = (z[idx] / m2) * spatial_lag
        I_values[idx] = I_i

        # Expected value under randomization
        E_Ii = -w_sum / (n - 1)

        # Variance under randomization (Anselin 1995 formula)
        # Var(Ii) = w_i2 * (n - b2) / (n - 1) + (w_i^2 - w_i2) * (2b2 - n) / ((n - 1)(n - 2)) - E(Ii)^2
        var_term1 = (w_sum2 * (n - b2)) / (n - 1)

        if n > 2:
            var_term2 = ((w_sum**2 - w_sum2) * (2*b2 - n)) / ((n - 1) * (n - 2))
        else:
            var_term2 = 0.0

        var_Ii = var_term1 + var_term2 - (E_Ii ** 2)

        if var_Ii > 0:
            z_i = (I_i - E_Ii) / math.sqrt(var_Ii)
            z_scores[idx] = z_i
            p = 1.0 - math.erf(abs(z_i) / math.sqrt(2.0))
            p_values[idx] = p

            # Quadrant categorization (HH, LL, HL, LH)
            if p < 0.05:
                # Value relative to mean
                high_val = z[idx] > 0
                # Lag relative to mean
                high_lag = spatial_lag > 0

                if high_val and high_lag:
                    quadrants[idx] = "HH"
                elif not high_val and not high_lag:
                    quadrants[idx] = "LL"
                elif high_val and not high_lag:
                    quadrants[idx] = "HL"
                elif not high_val and high_lag:
                    quadrants[idx] = "LH"
            else:
                quadrants[idx] = "Not Significant"
        else:
            z_scores[idx] = 0.0
            p_values[idx] = 1.0
            quadrants[idx] = "Not Significant"

    return I_values, z_scores, p_values, quadrants


def _chi2_sf_approx(x: float, df: int) -> float:
    """Wilson-Hilferty transformation approximation for Chi-Square Survival Function (p-value)."""
    if x <= 0:
        return 1.0
    if df == 2:
        return float(math.exp(-0.5 * x))  # Exact for df=2

    # Wilson-Hilferty approximation: Chi2 to normal
    d = float(df)
    z = ((x / d) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * d))) / math.sqrt(2.0 / (9.0 * d))
    p_val = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(max(0.0, min(1.0, p_val)))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) without scipy."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of Student's t with ``df`` degrees of freedom.

    P(|T| >= |t|) = I_{df / (df + t^2)}(df / 2, 1 / 2).
    """
    if not (math.isfinite(t) and df > 0):
        return float("nan")
    x = df / (df + t * t)
    return float(max(0.0, min(1.0, _betai(0.5 * df, 0.5, x))))


def calculate_ols(
    y: np.ndarray,
    X_data: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    x_names: list[str]
) -> dict:
    """Performs Ordinary Least Squares (OLS) regression and diagnostic tests.

    Args:
        y: 1D dependent variable array (n,)
        X_data: 2D independent variables array (n, p)
        neighbors: Weights neighbors dict
        weights: Weights values dict
        id_order: Feature IDs
        x_names: Names of independent variables

    Returns:
        A dictionary containing coefficient estimates, diagnostics, residuals, etc.
    """
    n = len(y)
    p = X_data.shape[1]

    # Add intercept column
    X = np.column_stack((np.ones(n), X_data))

    # Solve beta = (X.T * X)^-1 * X.T * Y
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
        beta = xtx_inv @ X.T @ y
    except Exception as e:
        logger.error("Linear algebra inversion failed in OLS regression: %s", e)
        raise ValueError(f"Regression inversion failed: {e}")

    # Residuals
    y_pred = X @ beta
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # Variance of residuals
    df_err = n - p - 1
    if df_err <= 0:
        raise ValueError(f"Sample size ({n}) must be greater than number of variables ({p} + intercept).")

    s2 = ss_res / df_err
    std_residuals = residuals / math.sqrt(s2) if s2 > 0 else np.zeros(n)

    # Standard Errors of Coefficients
    cov_beta = s2 * xtx_inv
    se_beta = np.sqrt(np.maximum(0.0, np.diagonal(cov_beta)))

    # t-statistics and p-values
    t_stats = np.zeros(p + 1)
    p_vals = np.ones(p + 1)
    for j in range(p + 1):
        if se_beta[j] > 0:
            t_stats[j] = beta[j] / se_beta[j]
            # Student t with the residual degrees of freedom (exact, not normal).
            p_vals[j] = student_t_two_sided_p(float(t_stats[j]), float(df_err))
        else:
            t_stats[j] = 0.0
            p_vals[j] = 1.0

    # Model R2 & Adj R2
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_err

    # --- DIAGNOSTIC 1: Jarque-Bera normality test ---
    s2_ml = ss_res / n
    if s2_ml > 0:
        skew = np.sum(residuals ** 3) / n / (s2_ml ** 1.5)
        kurt = np.sum(residuals ** 4) / n / (s2_ml ** 2)
        jb_stat = (n / 6.0) * (skew ** 2 + 0.25 * (kurt - 3.0) ** 2)
        jb_p = _chi2_sf_approx(jb_stat, df=2)
    else:
        jb_stat, jb_p = 0.0, 1.0

    # --- DIAGNOSTIC 2: Koenker's Breusch-Pagan heteroskedasticity test ---
    # Auxiliary regression: e^2 on X_data
    g = residuals ** 2
    g_mean = np.mean(g)
    g_tot = np.sum((g - g_mean) ** 2)

    bp_stat, bp_p = 0.0, 1.0
    if g_tot > 0:
        try:
            # Regress g on independent variables
            beta_aux = np.linalg.pinv(X.T @ X) @ X.T @ g
            g_pred = X @ beta_aux
            g_res = g - g_pred
            ss_aux_res = np.sum(g_res ** 2)
            r2_aux = 1.0 - (ss_aux_res / g_tot)
            bp_stat = n * r2_aux
            bp_p = _chi2_sf_approx(bp_stat, df=p)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            bp_stat, bp_p = 0.0, 1.0

    # --- DIAGNOSTIC 3: Moran's I on Residuals, I = (n / S0) e'We / e'e ---
    # NaN when the weights have no neighbour pair or the fit is exact: there
    # is nothing to measure, and 0 would read as "no autocorrelation".
    w_rows, w_cols, w_vals = weights_to_arrays(neighbors, weights, id_order)
    s0_res = float(np.sum(w_vals))
    if ss_res > 0 and s0_res != 0:
        cross = float(np.sum(w_vals * residuals[w_rows] * residuals[w_cols]))
        moran_i = (n / s0_res) * cross / ss_res
    else:
        moran_i = float("nan")

    # Return OLS results dictionary
    return {
        "coefficients": beta,
        "std_errors": se_beta,
        "t_statistics": t_stats,
        "p_values": p_vals,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "p": p,
        "df_err": df_err,
        "fitted": y_pred,
        "residuals": residuals,
        "std_residuals": std_residuals,
        "jarque_bera": (jb_stat, jb_p),
        "breusch_pagan": (bp_stat, bp_p),
        "residuals_moran": moran_i,
        "variable_names": ["Intercept"] + x_names
    }


def calculate_global_moran(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int]
) -> tuple[float, float, float, float, float]:
    """Calculates Global Moran's I spatial autocorrelation.

    Returns:
        A tuple of (moran_i, expected_i, variance, z_score, p_value)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= 3:
        raise _too_few("Global Moran's I", 4, n)
    _require_finite(y, "Global Moran's I")

    y_mean = np.mean(y)
    z = y - y_mean
    sum_z2 = np.sum(z ** 2)
    sum_z4 = np.sum(z ** 4)

    if sum_z2 == 0:
        raise _no_variation("Global Moran's I")

    rows, cols, vals = weights_to_arrays(neighbors, weights, id_order)
    S0, A1, A2, w_row_sums, w_col_sums = _sparse_weight_sums(rows, cols, vals, n)
    if S0 == 0:
        raise _no_neighbours("Global Moran's I")

    # S1 = 1/2 sum_ij (w_ij + w_ji)^2 over every pair, including pairs where
    # only w_ji is non-zero (asymmetric KNN weights): it equals A1 + A2.
    S1 = A1 + A2
    S2 = np.sum((w_row_sums + w_col_sums) ** 2)
    numerator = float(np.sum(vals * z[rows] * z[cols]))

    moran_i = (n / S0) * (numerator / sum_z2)

    # Expected value
    expected_i = -1.0 / (n - 1)

    # Kurtosis term D
    D = (n * sum_z4) / (sum_z2 ** 2)

    # Variance under randomization
    num_var = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2) - D * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    den_var = (n - 1) * (n - 2) * (n - 3) * S0 ** 2

    variance = num_var / den_var - (expected_i ** 2) if den_var > 0 else 0.0
    if not variance > 1e-12 * expected_i ** 2:
        raise InputError(
            "Every feature is a neighbour of every other feature, so Moran's I cannot differ "
            "from its expected value. Use fewer neighbours or a smaller distance band."
        )
    z_score = (moran_i - expected_i) / math.sqrt(variance)
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))

    return float(moran_i), float(expected_i), float(variance), float(z_score), float(p_value)


def calculate_spatial_gini(
    values: np.ndarray,
    neighbors: dict[int, list[int]],
    id_order: list[int],
    permutations: int = 99,
    seed: int = 42,
) -> dict:
    """Calculate classic Gini plus Rey-Smith style spatial Gini decomposition.

    The decomposition splits the pairwise absolute-difference numerator into
    neighbor and non-neighbor components. Components use the same denominator
    as the classic Gini, so neighbor_component + non_neighbor_component == gini.
    """
    y = np.array(values, dtype=float)
    finite = np.isfinite(y)
    if not np.all(finite):
        y = y[finite]
        id_order = [fid for fid, ok in zip(id_order, finite) if ok]

    n = int(len(y))
    if n < 2:
        raise ValueError("Spatial Gini requires at least 2 finite numeric observations.")
    if np.any(y < 0.0):
        raise ValueError("Gini coefficients require non-negative values.")

    mean_value = float(np.mean(y))
    pair_indices, neighbor_flags = _spatial_gini_pair_index(neighbors, id_order)
    total_pair_count = int(len(pair_indices))
    if total_pair_count == 0:
        raise ValueError("Spatial Gini requires at least one observation pair.")

    denominator = float((n ** 2) * mean_value)
    pair_sum, neighbor_sum, non_neighbor_sum = _spatial_gini_pair_sums(y, pair_indices, neighbor_flags)

    gini = pair_sum / denominator if denominator > 0.0 else 0.0
    neighbor_component = neighbor_sum / denominator if denominator > 0.0 else 0.0
    non_neighbor_component = non_neighbor_sum / denominator if denominator > 0.0 else 0.0

    neighbor_pair_count = int(sum(1 for flag in neighbor_flags if flag))
    non_neighbor_pair_count = total_pair_count - neighbor_pair_count
    neighbor_avg_diff = neighbor_sum / neighbor_pair_count if neighbor_pair_count else None
    non_neighbor_avg_diff = non_neighbor_sum / non_neighbor_pair_count if non_neighbor_pair_count else None
    neighbor_share = neighbor_sum / pair_sum if pair_sum > 0.0 else 0.0
    non_neighbor_share = non_neighbor_sum / pair_sum if pair_sum > 0.0 else 0.0
    polarization = None
    if (
        neighbor_avg_diff is not None
        and non_neighbor_avg_diff is not None
        and neighbor_avg_diff > 0.0
    ):
        polarization = non_neighbor_avg_diff / neighbor_avg_diff

    result = {
        "n": n,
        "mean": mean_value,
        "sum": float(np.sum(y)),
        "gini": float(gini),
        "pair_abs_sum": float(pair_sum),
        "neighbor_abs_sum": float(neighbor_sum),
        "non_neighbor_abs_sum": float(non_neighbor_sum),
        "neighbor_component": float(neighbor_component),
        "non_neighbor_component": float(non_neighbor_component),
        "neighbor_share": float(neighbor_share),
        "non_neighbor_share": float(non_neighbor_share),
        "spatial_gini": float(non_neighbor_share),
        "neighbor_pair_count": neighbor_pair_count,
        "non_neighbor_pair_count": non_neighbor_pair_count,
        "total_pair_count": total_pair_count,
        "neighbor_avg_diff": neighbor_avg_diff,
        "non_neighbor_avg_diff": non_neighbor_avg_diff,
        "polarization": polarization,
        "permutations": int(max(0, permutations)),
        "expected_non_neighbor_component": None,
        "std_non_neighbor_component": None,
        "z_non_neighbor_component": None,
        "p_sim": None,
        "p_low_sim": None,
        "expected_polarization": 1.0 if polarization is not None else None,
        "polarization_p_sim": None,
    }

    if permutations <= 0 or non_neighbor_pair_count == 0 or pair_sum <= 0.0:
        return result

    rng = np.random.default_rng(seed)
    sim_non_neighbor = np.zeros(int(permutations), dtype=float)
    sim_polarization = []
    for idx in range(int(permutations)):
        permuted = rng.permutation(y)
        sim_pair_sum, sim_neighbor_sum, sim_non_neighbor_sum = _spatial_gini_pair_sums(
            permuted,
            pair_indices,
            neighbor_flags,
        )
        if sim_pair_sum > 0.0:
            sim_non_neighbor[idx] = sim_non_neighbor_sum / denominator
        if neighbor_pair_count and non_neighbor_pair_count and sim_neighbor_sum > 0.0:
            sim_neighbor_avg = sim_neighbor_sum / neighbor_pair_count
            sim_non_neighbor_avg = sim_non_neighbor_sum / non_neighbor_pair_count
            sim_polarization.append(sim_non_neighbor_avg / sim_neighbor_avg)

    expected = float(np.mean(sim_non_neighbor))
    std = float(np.std(sim_non_neighbor))
    result["expected_non_neighbor_component"] = expected
    result["std_non_neighbor_component"] = std
    result["z_non_neighbor_component"] = (
        float((non_neighbor_component - expected) / std) if std > 0.0 else 0.0
    )
    result["p_sim"] = float((int(np.sum(sim_non_neighbor >= non_neighbor_component)) + 1) / (permutations + 1))
    result["p_low_sim"] = float((int(np.sum(sim_non_neighbor <= non_neighbor_component)) + 1) / (permutations + 1))

    if polarization is not None and sim_polarization:
        sim_pol = np.array(sim_polarization, dtype=float)
        result["polarization_p_sim"] = float((int(np.sum(sim_pol >= polarization)) + 1) / (len(sim_pol) + 1))

    return result


def _spatial_gini_pair_index(
    neighbors: dict[int, list[int]],
    id_order: list[int],
) -> tuple[list[tuple[int, int]], list[bool]]:
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    neighbor_pairs = set()
    for fid in id_order:
        i = id_to_idx[fid]
        for nid in neighbors.get(fid, []):
            if nid not in id_to_idx or nid == fid:
                continue
            j = id_to_idx[nid]
            neighbor_pairs.add((i, j) if i < j else (j, i))

    pair_indices: list[tuple[int, int]] = []
    neighbor_flags: list[bool] = []
    n = len(id_order)
    for i in range(n - 1):
        for j in range(i + 1, n):
            pair = (i, j)
            pair_indices.append(pair)
            neighbor_flags.append(pair in neighbor_pairs)
    return pair_indices, neighbor_flags


def _spatial_gini_pair_sums(
    y: np.ndarray,
    pair_indices: list[tuple[int, int]],
    neighbor_flags: list[bool],
) -> tuple[float, float, float]:
    total_sum = 0.0
    neighbor_sum = 0.0
    non_neighbor_sum = 0.0
    for (i, j), is_neighbor in zip(pair_indices, neighbor_flags):
        diff = abs(float(y[i]) - float(y[j]))
        total_sum += diff
        if is_neighbor:
            neighbor_sum += diff
        else:
            non_neighbor_sum += diff
    return total_sum, neighbor_sum, non_neighbor_sum


def calculate_average_nearest_neighbor(
    x: np.ndarray,
    y: np.ndarray,
    study_area: Optional[float] = None
) -> tuple[float, float, float, float, float, float]:
    """Calculates Average Nearest Neighbor statistics.

    Returns:
        A tuple of (observed_mean, expected_mean, nn_ratio, z_score, p_value, study_area)
    """
    n = len(x)
    if n <= 1:
        raise ValueError("Average Nearest Neighbor requires at least 2 points.")

    coords = np.column_stack((x, y))

    # Try using scikit-learn KDTree for maximum performance, fallback to vectorized NumPy
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        nn_dists = distances[:, 1]
    except ImportError:
        # Vectorized chunked NumPy distance finder to protect memory
        nn_dists = np.zeros(n)
        chunk_size = 1000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_coords = coords[start:end]
            d = np.sqrt(((chunk_coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
            for i in range(start, end):
                d[i - start, i] = np.inf
            nn_dists[start:end] = np.min(d, axis=1)

    observed_mean = float(np.mean(nn_dists))

    # Fallback to minimum bounding box area if study area is not provided
    if study_area is None or study_area <= 0:
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        w = max_x - min_x
        h = max_y - min_y
        study_area = float(w * h) if (w * h > 0) else 1.0

    density = n / study_area
    expected_mean = 0.5 / math.sqrt(density)

    # Standard error
    se = 0.26136 / math.sqrt(n * density)

    nn_ratio = observed_mean / expected_mean if expected_mean > 0 else 1.0
    z_score = (observed_mean - expected_mean) / se if se > 0 else 0.0
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

    return observed_mean, expected_mean, nn_ratio, z_score, p_value, study_area


def calculate_standard_distance(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> tuple[float, float, float]:
    """Calculates Standard Distance and mean center.

    Returns:
        A tuple of (mean_x, mean_y, standard_distance)
    """
    n = len(x_coords)
    if n == 0:
        return 0.0, 0.0, 0.0

    mean_x, mean_y = calculate_mean_center(x_coords, y_coords, weights)

    if weights is None or len(weights) == 0:
        var_x = np.sum((x_coords - mean_x) ** 2) / n
        var_y = np.sum((y_coords - mean_y) ** 2) / n
    else:
        sum_w = np.sum(weights)
        if sum_w == 0:
            var_x = np.sum((x_coords - mean_x) ** 2) / n
            var_y = np.sum((y_coords - mean_y) ** 2) / n
        else:
            var_x = np.sum(weights * (x_coords - mean_x) ** 2) / sum_w
            var_y = np.sum(weights * (y_coords - mean_y) ** 2) / sum_w

    std_distance = math.sqrt(var_x + var_y)
    return mean_x, mean_y, std_distance


def calculate_gwr(
    y: np.ndarray,
    X_data: np.ndarray,
    coords: np.ndarray,
    bandwidth: float,
    kernel_type: str = "adaptive_bisquare",
    leave_one_out: bool = False,
) -> dict:
    """Performs Geographically Weighted Regression (GWR) analysis.

    Args:
        y: Dependent variable (n,)
        X_data: Independent variables (n, p)
        coords: Centroid coordinates (n, 2)
        bandwidth: Kernel bandwidth (distance or neighbor count)
        kernel_type: fixed_gaussian, fixed_bisquare, or adaptive_bisquare

    Returns:
        A dictionary containing GWR coefficients, errors, local R2, and statistics.
    """
    n = len(y)
    p = X_data.shape[1]

    # Add intercept column
    X = np.column_stack((np.ones(n), X_data))

    # Initialize results
    local_beta = np.zeros((n, p + 1))
    local_se = np.zeros((n, p + 1))
    local_t = np.zeros((n, p + 1))
    y_pred = np.zeros(n)
    local_r2 = np.zeros(n)
    local_support = np.zeros(n, dtype=int)

    # Compute distance matrix
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    for i in range(n):
        dists = dists_matrix[i]

        # Calculate weights based on kernel type
        if kernel_type == "fixed_gaussian":
            w = np.exp(-0.5 * (dists / bandwidth) ** 2)
        elif kernel_type == "fixed_bisquare":
            w = np.zeros(n)
            mask = dists < bandwidth
            w[mask] = (1.0 - (dists[mask] / bandwidth) ** 2) ** 2
        elif kernel_type == "adaptive_bisquare":
            k = int(bandwidth)
            sorted_dists = np.sort(dists)
            d_k = sorted_dists[min(k - 1, n - 1)]
            w = np.zeros(n)
            if d_k > 0:
                mask = dists < d_k
                w[mask] = (1.0 - (dists[mask] / d_k) ** 2) ** 2
            else:
                w[dists == 0] = 1.0
        else:
            w = np.ones(n)

        if leave_one_out:
            w[i] = 0.0

        sum_w = np.sum(w)
        if sum_w == 0:
            w = np.ones(n)
            sum_w = n
        local_support[i] = int(np.sum(w > 1e-12))

        # Solve local regression: beta_i = (X.T * W * X)^-1 * X.T * W * Y
        try:
            xtw = X.T * w
            xtwx = xtw @ X
            xtwx_inv = np.linalg.pinv(xtwx)
            beta_i = xtwx_inv @ xtw @ y
            local_beta[i] = beta_i
            y_pred[i] = X[i] @ beta_i

            # Standard errors and t-statistics
            res_i = y - (X @ beta_i)
            df_i = sum_w - p - 1
            if df_i > 0:
                s2_i = np.sum(w * (res_i ** 2)) / df_i
                cov_beta_i = s2_i * xtwx_inv
                se_beta_i = np.sqrt(np.maximum(0.0, np.diagonal(cov_beta_i)))
                local_se[i] = se_beta_i
                for j in range(p + 1):
                    if se_beta_i[j] > 0:
                        local_t[i, j] = beta_i[j] / se_beta_i[j]

            # Local R2
            y_w_mean = np.sum(w * y) / sum_w
            tss_i = np.sum(w * (y - y_w_mean) ** 2)
            rss_i = np.sum(w * (res_i ** 2))
            local_r2[i] = 1.0 - (rss_i / tss_i) if tss_i > 0 else 1.0
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            local_r2[i] = np.nan

    residuals = y - y_pred
    rss = np.sum(residuals ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    global_r2 = 1.0 - (rss / tss) if tss > 0 else 0.0

    # Effective degrees of freedom (Trace of Hat Matrix)
    tr_S = 0.0
    for i in range(n):
        try:
            if kernel_type == "fixed_gaussian":
                w = np.exp(-0.5 * (dists_matrix[i] / bandwidth) ** 2)
            elif kernel_type == "fixed_bisquare":
                w = np.zeros(n)
                mask = dists_matrix[i] < bandwidth
                w[mask] = (1.0 - (dists_matrix[i][mask] / bandwidth) ** 2) ** 2
            elif kernel_type == "adaptive_bisquare":
                k = int(bandwidth)
                d_k = np.sort(dists_matrix[i])[min(k - 1, n - 1)]
                w = np.zeros(n)
                if d_k > 0:
                    mask = dists_matrix[i] < d_k
                    w[mask] = (1.0 - (dists_matrix[i][mask] / d_k) ** 2) ** 2
                else:
                    w[dists_matrix[i] == 0] = 1.0
            else:
                w = np.ones(n)

            xtw = X.T * w
            xtwx_inv = np.linalg.pinv(xtw @ X)
            s_i = (X[i] @ xtwx_inv) @ xtw
            tr_S += s_i[i]
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, IndexError):
            tr_S += 0.0

    if tr_S <= 0:
        tr_S = float(p + 1)

    aicc = np.inf
    if n - tr_S - 2 > 0 and rss > 0:
        aicc = n * np.log(rss / n) + n * np.log(2 * np.pi) + n * (n + tr_S) / (n - 2 - tr_S)

    return {
        "local_beta": local_beta,
        "local_se": local_se,
        "local_t": local_t,
        "local_support": local_support,
        "y_pred": y_pred,
        "residuals": residuals,
        "local_r2": local_r2,
        "rss": rss,
        "tss": tss,
        "r2": global_r2,
        "aicc": aicc,
        "effective_df": tr_S,
        "cv_rmse": float(np.sqrt(np.mean(residuals ** 2))),
    }


def select_gwr_bandwidth(
    y: np.ndarray,
    X_data: np.ndarray,
    coords: np.ndarray,
    kernel_type: str,
    criterion: str = "aicc",
    candidate_count: int = 8,
) -> dict:
    """Select a defensible GWR bandwidth by AICc or leave-one-out CV."""
    n = len(y)
    minimum_support = min(n - 1, max(8, X_data.shape[1] + 3))
    if kernel_type == "adaptive_bisquare":
        candidates = np.unique(
            np.linspace(minimum_support, max(minimum_support, n - 1), candidate_count).astype(int)
        ).astype(float)
    else:
        distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
        nonzero = distances[distances > 0]
        if nonzero.size == 0:
            raise ValueError("Automatic GWR bandwidth selection needs distinct coordinates.")
        candidates = np.unique(np.quantile(nonzero, np.linspace(0.10, 0.90, candidate_count)))

    trials = []
    for bandwidth in candidates:
        result = calculate_gwr(
            y,
            X_data,
            coords,
            float(bandwidth),
            kernel_type,
            leave_one_out=(criterion == "cv"),
        )
        score = float(result["cv_rmse"] if criterion == "cv" else result["aicc"])
        trials.append({"bandwidth": float(bandwidth), "score": score})
    finite = [trial for trial in trials if np.isfinite(trial["score"])]
    if not finite:
        raise ValueError("No finite score was produced during GWR bandwidth selection.")
    best = min(finite, key=lambda trial: trial["score"])
    return {
        "bandwidth": best["bandwidth"],
        "criterion": criterion,
        "score": best["score"],
        "trials": trials,
    }


def calculate_median_center(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6
) -> tuple[float, float, float]:
    """Calculates Median Center using Weiszfeld's algorithm.

    Returns:
        A tuple of (median_x, median_y, total_distance)
    """
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return float(x[0]), float(y[0]), 0.0

    if weights is None or len(weights) == 0:
        weights = np.ones(n)

    # Initial guess: mean center
    cx, cy = calculate_mean_center(x, y, weights)

    for _ in range(max_iter):
        dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        dists = np.maximum(dists, 1e-12)

        inv_dists = weights / dists
        sum_inv = np.sum(inv_dists)

        if sum_inv == 0:
            break

        new_cx = np.sum(x * inv_dists) / sum_inv
        new_cy = np.sum(y * inv_dists) / sum_inv

        shift = math.sqrt((new_cx - cx) ** 2 + (new_cy - cy) ** 2)
        cx, cy = new_cx, new_cy

        if shift < tol:
            break

    total_dist = float(np.sum(weights * np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))
    return float(cx), float(cy), total_dist


def general_g_from_arrays(x, rows, cols, vals) -> tuple[float, float, float, float, float]:
    """Getis-Ord General G under randomisation from sparse weights (Getis & Ord 1992).

    ``rows``/``cols``/``vals`` list the non-zero w_ij (i != j, each pair once)
    over the positions of ``x``. Memory and time are O(n + nnz log nnz); no
    n x n matrix is built. With A1 = sum w_ij^2, A2 = sum w_ij w_ji,
    A3 = sum_i r_i^2 + sum_j c_j^2, A4 = sum_i r_i c_i, A5 = S0^2 and the
    distinct-index moment sums S22, S211, S1111 of x,
        E[(sum w_ij x_i x_j)^2] = (A1 + A2) S22 / n(2)
                                + (A3 + 2 A4 - 2 A1 - 2 A2) S211 / n(3)
                                + (A5 - A3 - 2 A4 + A1 + A2) S1111 / n(4)
    where n(k) = n (n - 1) ... (n - k + 1); this equals esda.G's VG for binary
    weights and the exact permutation variance for any weights.

    Returns (observed_g, expected_g, variance, z_score, p_value).
    Raises InputError for fewer than 4 features, missing values, a constant
    field, a field with a single non-zero value, or weights with S0 = 0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        raise _too_few("General G", 4, n)
    _require_finite(x, "General G")
    if float(np.ptp(x)) == 0.0:
        raise _no_variation("General G")
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=float)
    off = rows != cols
    rows, cols, vals = rows[off], cols[off], vals[off]
    s0 = float(np.sum(vals))
    if vals.size == 0 or s0 == 0.0:
        raise _no_neighbours("General G")

    s1, s2 = float(np.sum(x)), float(np.sum(x ** 2))
    s3, s4 = float(np.sum(x ** 3)), float(np.sum(x ** 4))
    denominator = s1 ** 2 - s2
    if denominator == 0.0:
        raise InputError(
            "General G needs at least two features with a non-zero value; "
            "the analysis field has fewer."
        )

    _, a1, a2, row_sums, col_sums = _sparse_weight_sums(rows, cols, vals, n)
    a3 = float(np.sum(row_sums ** 2) + np.sum(col_sums ** 2))
    a4 = float(np.sum(row_sums * col_sums))
    a5 = s0 ** 2

    observed_g = float(np.sum(vals * x[rows] * x[cols])) / denominator
    expected_g = s0 / (n * (n - 1))

    s22 = s2 ** 2 - s4
    s211 = s2 * (s1 ** 2 - s2) - 2.0 * s1 * s3 + 2.0 * s4
    s1111 = s1 ** 4 - 6.0 * s1 ** 2 * s2 + 8.0 * s1 * s3 + 3.0 * s2 ** 2 - 6.0 * s4

    term1 = (a1 + a2) * s22 / (n * (n - 1))
    term2 = (a3 + 2.0 * a4 - 2.0 * a1 - 2.0 * a2) * s211 / (n * (n - 1) * (n - 2))
    term3 = (a5 - a3 - 2.0 * a4 + a1 + a2) * s1111 / (n * (n - 1) * (n - 2) * (n - 3))
    variance = (term1 + term2 + term3) / denominator ** 2 - expected_g ** 2

    if not variance > 1e-12 * expected_g ** 2:
        # Every permutation gives the same G: typically every feature is a
        # neighbour of every other one (a band wider than the study area).
        raise InputError(
            "Every feature is a neighbour of every other feature, so General G cannot differ "
            "from its expected value. Use a smaller distance band or fewer neighbours."
        )
    z_score = (observed_g - expected_g) / math.sqrt(variance)
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return float(observed_g), float(expected_g), float(variance), float(z_score), float(p_value)


def distance_band_pairs(coords, distance_band: float) -> tuple[np.ndarray, np.ndarray]:
    """All ordered pairs (i, j), i != j, with Euclidean distance <= ``distance_band``.

    A uniform grid of cell size ``distance_band`` limits each point's search to
    its own and the eight surrounding cells, so the cost is O(n + pairs)
    instead of O(n^2). Returns (rows, cols) index arrays, sorted by row.
    """
    pts = np.asarray(coords, dtype=float).reshape(-1, 2)
    n = pts.shape[0]
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if n < 2 or not (distance_band > 0) or not math.isfinite(distance_band):
        return empty
    if not np.all(np.isfinite(pts)):
        raise InputError("Some feature centroids have no finite coordinates; repair the geometries first.")
    cell = np.floor((pts - pts.min(axis=0)) / float(distance_band)).astype(np.int64)
    # Cell key with a one-cell margin so that +/-1 offsets never wrap.
    width = int(cell[:, 1].max()) + 3
    key = (cell[:, 0] + 1) * width + (cell[:, 1] + 1)
    order = np.argsort(key, kind="mergesort")
    skey = key[order]
    band2 = float(distance_band) ** 2
    out_r = []
    out_c = []
    # Points are handled in chunks so the candidate arrays stay bounded.
    chunk = 200000
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            target = key + dx * width + dy
            lo = np.searchsorted(skey, target, side="left")
            hi = np.searchsorted(skey, target, side="right")
            counts = hi - lo
            for s in range(0, n, chunk):
                cnt = counts[s:s + chunk]
                total = int(cnt.sum())
                if total == 0:
                    continue
                src = np.repeat(np.arange(s, min(s + chunk, n)), cnt)
                offs = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                dst = order[np.repeat(lo[s:s + chunk], cnt) + offs]
                d2 = ((pts[src] - pts[dst]) ** 2).sum(axis=1)
                keep = (d2 <= band2) & (src != dst)
                out_r.append(src[keep])
                out_c.append(dst[keep])
    if not out_r:
        return empty
    rows = np.concatenate(out_r).astype(np.int64)
    cols = np.concatenate(out_c).astype(np.int64)
    srt = np.lexsort((cols, rows))
    return rows[srt], cols[srt]


def calculate_general_g_distance_band(values, coords, distance_band: float) -> tuple[float, float, float, float, float]:
    """General G with binary distance-band weights from point coordinates, sparse end to end.

    ``values`` and ``coords`` (n x 2, projected units) share one order.
    Same return value and errors as ``general_g_from_arrays``.
    """
    x = np.asarray(values, dtype=float)
    rows, cols = distance_band_pairs(coords, distance_band)
    return general_g_from_arrays(x, rows, cols, np.ones(rows.size))


def calculate_general_g(
    values,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int]
) -> tuple[float, float, float, float, float]:
    """Calculates Getis-Ord General G statistics under randomization.

    ``values`` is a dict keyed by feature id or a sequence in ``id_order``.
    Weights are used as given (binary in the General G tool). See
    ``general_g_from_arrays`` for the formula and the errors raised.

    Returns:
        A tuple of (observed_g, expected_g, variance, z_score, p_value)
    """
    n = len(id_order)
    if n < 4:
        raise _too_few("General G", 4, n)
    if isinstance(values, dict):
        x = np.array([float(values[fid]) for fid in id_order])
    else:
        x = np.array([float(values[i]) for i in range(n)])
    rows, cols, vals = weights_to_arrays(neighbors, weights, id_order)
    return general_g_from_arrays(x, rows, cols, vals)


def calculate_similarity_search(
    full_data: np.ndarray,
    target_indices: list[int],
    metric: str = "euclidean"
) -> np.ndarray:
    """Standardizes attributes and computes distance score from target feature profiles.

    Returns:
        An array of similarity scores for each feature in full_data.
    """
    n, p = full_data.shape
    if n == 0 or p == 0:
        return np.array([])

    # Z-score standardization
    means = np.mean(full_data, axis=0)
    stds = np.std(full_data, axis=0)
    stds[stds == 0.0] = 1.0  # avoid division by zero

    z_data = (full_data - means) / stds

    # Extract target profile (mean profile if multiple targets are selected)
    z_targets = z_data[target_indices]
    target_profile = np.mean(z_targets, axis=0)

    # Compute score based on selected distance metric
    if metric == "manhattan":
        scores = np.sum(np.abs(z_data - target_profile), axis=1)
    else:  # euclidean
        scores = np.sqrt(np.sum((z_data - target_profile) ** 2, axis=1))

    return scores


def calculate_distance_band_stats(
    x: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 1
) -> dict:
    """Calculates statistics for distance to the k-th nearest neighbor.

    Returns:
        A dictionary containing min, max, mean, median, p25, p75 values.
    """
    n = len(x)
    if n <= 1:
        raise ValueError("At least 2 points are required to compute distance bands.")

    coords = np.column_stack((x, y))

    # Compute distance matrix
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    k_dists = np.zeros(n)
    for i in range(n):
        sorted_d = np.sort(dists_matrix[i])
        k_idx = min(k_neighbors, n - 1)
        k_dists[i] = sorted_d[k_idx]

    return {
        "min": float(np.min(k_dists)),
        "max": float(np.max(k_dists)),
        "mean": float(np.mean(k_dists)),
        "median": float(np.median(k_dists)),
        "p25": float(np.percentile(k_dists, 25)),
        "p75": float(np.percentile(k_dists, 75))
    }


def calculate_kmeans(
    data: np.ndarray,
    k_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42
) -> tuple[np.ndarray, float]:
    """Performs K-Means clustering on feature attribute data.

    Returns:
        A tuple of (labels, wcss) where wcss is within-cluster sum of squares.
    """
    n, p = data.shape
    if n < k_clusters:
        raise ValueError("Number of data points must be greater than or equal to k_clusters.")

    # Z-score standardization
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0.0] = 1.0
    z_data = (data - means) / stds

    # K-Means++ initialization
    rng = np.random.default_rng(seed)
    centroids = np.zeros((k_clusters, p))

    # Pick first centroid
    idx = rng.choice(n)
    centroids[0] = z_data[idx]

    for c_idx in range(1, k_clusters):
        # Distance squared to closest centroid
        dists_sq = np.min([np.sum((z_data - centroids[c]) ** 2, axis=1) for c in range(c_idx)], axis=0)

        # Avoid division by zero if all points are at the centroids
        sum_dists = np.sum(dists_sq)
        if sum_dists == 0:
            probs = np.ones(n) / n
        else:
            probs = dists_sq / sum_dists

        idx = rng.choice(n, p=probs)
        centroids[c_idx] = z_data[idx]

    # Lloyd's algorithm iterations
    labels = np.zeros(n, dtype=int)
    prev_wcss = np.inf

    for _ in range(max_iter):
        # Assign labels
        dists = np.array([np.sum((z_data - centroids[c]) ** 2, axis=1) for c in range(k_clusters)])
        labels = np.argmin(dists, axis=0)

        # Calculate WCSS
        wcss = 0.0
        for c in range(k_clusters):
            mask = (labels == c)
            if np.any(mask):
                wcss += np.sum((z_data[mask] - centroids[c]) ** 2)

        if abs(prev_wcss - wcss) < tol:
            break
        prev_wcss = wcss

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for c in range(k_clusters):
            mask = (labels == c)
            if np.any(mask):
                new_centroids[c] = np.mean(z_data[mask], axis=0)
            else:
                # Re-initialize empty cluster with a random point
                new_centroids[c] = z_data[rng.choice(n)]
        centroids = new_centroids

    return labels, float(wcss)


def calculate_linear_directional_mean(
    start_x: np.ndarray,
    start_y: np.ndarray,
    end_x: np.ndarray,
    end_y: np.ndarray
) -> tuple[float, float, float, float]:
    """Calculates Linear Directional Mean for line features.

    Returns:
        A tuple of (center_x, center_y, mean_angle_degrees, mean_length)
    """
    n = len(start_x)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Line midpoints
    mid_x = (start_x + end_x) / 2.0
    mid_y = (start_y + end_y) / 2.0

    center_x = float(np.mean(mid_x))
    center_y = float(np.mean(mid_y))

    # Line lengths
    dx = end_x - start_x
    dy = end_y - start_y
    lengths = np.sqrt(dx ** 2 + dy ** 2)
    lengths_safe = np.maximum(lengths, 1e-12)

    mean_length = float(np.mean(lengths))

    # Orientation angles (radians, compass-style: 0=North, clockwise)
    # atan2(dx, dy) gives compass bearing
    angles = np.arctan2(dx, dy)

    # Circular mean weighted by length
    sin_sum = np.sum(lengths_safe * np.sin(angles))
    cos_sum = np.sum(lengths_safe * np.cos(angles))

    mean_angle_rad = math.atan2(sin_sum, cos_sum)
    mean_angle_deg = math.degrees(mean_angle_rad) % 360.0

    return center_x, center_y, mean_angle_deg, mean_length


def _compute_moran_i_fast(
    z: np.ndarray,
    W: np.ndarray,
    S0: float,
    n: int
) -> float:
    """Fast internal Moran's I calculator using pre-built weight matrix."""
    sum_z2 = np.sum(z ** 2)
    if sum_z2 == 0:
        return 0.0
    numerator = float(z @ W @ z)
    return (n / S0) * (numerator / sum_z2)


def run_sensitivity_simulation(
    values: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    n_simulations: int = 999,
    seed: int = 42
) -> dict:
    """Monte Carlo simulation for Global Moran's I sensitivity assessment.

    Returns:
        A dictionary with observed_i, simulated_mean, simulated_std,
        empirical_p, percentile_5, percentile_95, and simulated_values.
    """
    n = len(id_order)
    if n < 4:
        raise ValueError("Sensitivity simulation requires at least 4 features.")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    # Build dense weight matrix
    W = np.zeros((n, n))
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                W[i, j] = w
    np.fill_diagonal(W, 0.0)
    S0 = float(np.sum(W))

    if S0 == 0:
        raise ValueError("No spatial neighbors found. Cannot run simulation.")

    # Observed Moran's I
    y = np.array([float(values[i]) for i in range(n)])
    z_obs = y - np.mean(y)
    observed_i = _compute_moran_i_fast(z_obs, W, S0, n)

    # Monte Carlo permutations
    rng = np.random.default_rng(seed)
    sim_values = np.zeros(n_simulations)

    for s in range(n_simulations):
        y_perm = rng.permutation(y)
        z_perm = y_perm - np.mean(y_perm)
        sim_values[s] = _compute_moran_i_fast(z_perm, W, S0, n)

    # Empirical p-value (two-tailed)
    count_extreme = np.sum(np.abs(sim_values) >= abs(observed_i))
    empirical_p = float((count_extreme + 1) / (n_simulations + 1))

    return {
        "observed_i": float(observed_i),
        "simulated_mean": float(np.mean(sim_values)),
        "simulated_std": float(np.std(sim_values)),
        "empirical_p": empirical_p,
        "percentile_5": float(np.percentile(sim_values, 5)),
        "percentile_95": float(np.percentile(sim_values, 95)),
        "simulated_values": sim_values.tolist()
    }


def calculate_incremental_autocorrelation(
    x: np.ndarray,
    y_coords: np.ndarray,
    values: np.ndarray,
    start_dist: float,
    dist_increment: float,
    n_increments: int
) -> list[dict]:
    """Calculates Global Moran's I at multiple distance bands.

    Returns:
        A list of dicts, each with keys: distance, morans_i, expected_i, z_score, p_value.
    """
    n = len(x)
    if n < 4:
        raise ValueError("Incremental autocorrelation requires at least 4 features.")

    coords = np.column_stack((x, y_coords))
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    y_mean = np.mean(values)
    z = values - y_mean
    sum_z2 = np.sum(z ** 2)

    if sum_z2 == 0:
        return [{"distance": start_dist + i * dist_increment,
                 "morans_i": 0.0, "expected_i": -1.0 / (n - 1),
                 "z_score": 0.0, "p_value": 1.0}
                for i in range(n_increments)]

    results = []
    for inc in range(n_increments):
        threshold = start_dist + inc * dist_increment

        W = (dists_matrix <= threshold).astype(float)
        np.fill_diagonal(W, 0.0)

        S0 = np.sum(W)
        neighbor_counts = np.sum(W > 0, axis=1)
        min_neighbors = int(np.min(neighbor_counts))
        median_neighbors = float(np.median(neighbor_counts))
        max_neighbors = int(np.max(neighbor_counts))
        isolated_count = int(np.sum(neighbor_counts == 0))
        if S0 == 0:
            results.append({
                "distance": threshold,
                "morans_i": 0.0,
                "expected_i": -1.0 / (n - 1),
                "z_score": 0.0,
                "p_value": 1.0,
                "min_neighbors": min_neighbors,
                "median_neighbors": median_neighbors,
                "max_neighbors": max_neighbors,
                "isolated_count": isolated_count,
            })
            continue

        numerator = float(z @ W @ z)
        morans_i = (n / S0) * (numerator / sum_z2)
        expected_i = -1.0 / (n - 1)

        # Variance under randomization (simplified)
        S1 = float(np.sum((W + W.T) ** 2)) / 2.0
        row_sums = np.sum(W, axis=1)
        col_sums = np.sum(W, axis=0)
        S2 = float(np.sum((row_sums + col_sums) ** 2))

        sum_z4 = np.sum(z ** 4)
        D = (n * sum_z4) / (sum_z2 ** 2)

        num_var = (n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
                   - D * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2))
        den_var = (n - 1) * (n - 2) * (n - 3) * S0 ** 2

        variance = num_var / den_var - (expected_i ** 2) if den_var > 0 else 0.0

        if variance > 0:
            z_score = (morans_i - expected_i) / math.sqrt(variance)
            p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
        else:
            z_score = 0.0
            p_value = 1.0

        results.append({
            "distance": threshold,
            "morans_i": float(morans_i),
            "expected_i": float(expected_i),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "min_neighbors": min_neighbors,
            "median_neighbors": median_neighbors,
            "max_neighbors": max_neighbors,
            "isolated_count": isolated_count,
        })

    return results


def calculate_ripleys_k(
    x: np.ndarray,
    y_coords: np.ndarray,
    start_dist: float,
    dist_increment: float,
    n_increments: int,
    study_area: Optional[float] = None
) -> list[dict]:
    """Calculates Ripley's K, expected K, and L(d)-d across distance bands."""
    n = len(x)
    if n < 3:
        raise ValueError("Ripley's K requires at least 3 features.")
    coords = np.column_stack((x, y_coords))
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    if study_area is None or study_area <= 0:
        width = float(np.max(x) - np.min(x))
        height = float(np.max(y_coords) - np.min(y_coords))
        study_area = max(width * height, 1e-12)

    results = []
    for inc in range(n_increments):
        distance = start_dist + inc * dist_increment
        within = (dists_matrix <= distance).astype(float)
        np.fill_diagonal(within, 0.0)
        observed_pairs = float(np.sum(within))
        observed_k = (study_area / (n * (n - 1))) * observed_pairs
        expected_k = math.pi * (distance ** 2)
        l_value = math.sqrt(max(observed_k, 0.0) / math.pi) if observed_k >= 0 else 0.0
        l_minus_d = l_value - distance
        neighbor_counts = np.sum(within > 0, axis=1)
        results.append({
            "distance": float(distance),
            "observed_k": float(observed_k),
            "expected_k": float(expected_k),
            "l_value": float(l_value),
            "l_minus_d": float(l_minus_d),
            "observed_pairs": int(observed_pairs),
            "min_neighbors": int(np.min(neighbor_counts)),
            "median_neighbors": float(np.median(neighbor_counts)),
            "max_neighbors": int(np.max(neighbor_counts)),
            "isolated_count": int(np.sum(neighbor_counts == 0)),
            "study_area": float(study_area),
        })
    return results


def calculate_exploratory_regression(
    y: np.ndarray,
    X_data: np.ndarray,
    x_names: list[str],
    max_vars: Optional[int] = None
) -> list[dict]:
    """Tests all possible OLS variable combinations and returns ranked models.

    Returns:
        A sorted list of model dicts with keys: variables, r2, adj_r2, aic, coefficients.
    """
    from itertools import combinations

    n, total_p = X_data.shape
    if max_vars is None:
        max_vars = min(total_p, 5)  # cap at 5 to keep runtime manageable
    max_vars = min(max_vars, total_p)

    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return []

    models = []

    for k in range(1, max_vars + 1):
        for combo in combinations(range(total_p), k):
            X_sub = X_data[:, combo]
            X_design = np.column_stack((np.ones(n), X_sub))

            try:
                xtx_inv = np.linalg.pinv(X_design.T @ X_design)
                beta = xtx_inv @ X_design.T @ y
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue

            y_pred = X_design @ beta
            residuals = y - y_pred
            ss_res = np.sum(residuals ** 2)

            p = len(combo)
            df = n - p - 1
            if df <= 0:
                continue

            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / df) if df > 0 else 0.0

            # AICc
            if ss_res > 0 and n > 0:
                log_lik = -n / 2.0 * (np.log(2.0 * np.pi * ss_res / n) + 1.0)
                k_params = p + 2  # intercept + vars + variance
                aic = -2.0 * log_lik + 2.0 * k_params
                if n - k_params - 1 > 0:
                    aicc = aic + (2.0 * k_params * (k_params + 1)) / (n - k_params - 1)
                else:
                    aicc = float('inf')
            else:
                aicc = float('inf')

            var_names = [x_names[i] for i in combo]
            coef_dict = {"Intercept": float(beta[0])}
            for idx, vi in enumerate(combo):
                coef_dict[x_names[vi]] = float(beta[idx + 1])

            models.append({
                "variables": var_names,
                "r2": float(r2),
                "adj_r2": float(adj_r2),
                "aicc": float(aicc),
                "coefficients": coef_dict,
                "n_vars": p
            })

    # Sort by AICc ascending (best first)
    models.sort(key=lambda m: m["aicc"])
    return models


def calculate_glr(
    y: np.ndarray,
    X_data: np.ndarray,
    family: str = "gaussian",
    max_iter: int = 100,
    tol: float = 1e-6
) -> dict:
    """Fits Gaussian, logistic, or Poisson generalized linear regression."""
    n = len(y)
    p = X_data.shape[1]
    if n <= p + 1:
        raise ValueError("GLR requires more observations than model parameters.")
    X = np.column_stack((np.ones(n), X_data))
    family = family.lower()

    if family == "gaussian":
        xtx_inv = np.linalg.pinv(X.T @ X)
        beta = xtx_inv @ X.T @ y
        mu = X @ beta
        residuals = y - mu
        rss = float(np.sum(residuals ** 2))
        df = n - p - 1
        sigma2 = rss / df if df > 0 else 0.0
        cov = sigma2 * xtx_inv
        se = np.sqrt(np.maximum(0.0, np.diagonal(cov)))
        z_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        p_values = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(z_stats) / math.sqrt(2.0))))
        tss = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (rss / tss) if tss > 0 else 0.0
        log_likelihood = -n / 2.0 * (math.log(2.0 * math.pi * rss / n) + 1.0) if rss > 0 else 0.0
        aic = -2.0 * log_likelihood + 2.0 * (p + 2)
        return {
            "family": "gaussian",
            "coefficients": beta,
            "std_errors": se,
            "z_statistics": z_stats,
            "p_values": p_values,
            "fitted": mu,
            "residuals": residuals,
            "log_likelihood": float(log_likelihood),
            "aic": float(aic),
            "r2": float(r2),
            "iterations": 1,
            "converged": True,
        }

    if family == "logistic":
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Logistic GLR requires a binary dependent variable coded as 0 and 1.")
        beta = np.zeros(p + 1)
        converged = False
        for iteration in range(1, max_iter + 1):
            eta = X @ beta
            mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
            w = np.maximum(mu * (1.0 - mu), 1e-9)
            z = eta + (y - mu) / w
            xtw = X.T * w
            new_beta = np.linalg.pinv(xtw @ X) @ xtw @ z
            if np.max(np.abs(new_beta - beta)) < tol:
                beta = new_beta
                converged = True
                break
            beta = new_beta
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
        w = np.maximum(mu * (1.0 - mu), 1e-9)
        cov = np.linalg.pinv((X.T * w) @ X)
        se = np.sqrt(np.maximum(0.0, np.diagonal(cov)))
        z_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        p_values = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(z_stats) / math.sqrt(2.0))))
        eps = 1e-12
        log_likelihood = float(np.sum(y * np.log(mu + eps) + (1.0 - y) * np.log(1.0 - mu + eps)))
        aic = -2.0 * log_likelihood + 2.0 * (p + 1)
        residuals = y - mu
        return {
            "family": "logistic",
            "coefficients": beta,
            "std_errors": se,
            "z_statistics": z_stats,
            "p_values": p_values,
            "fitted": mu,
            "residuals": residuals,
            "log_likelihood": log_likelihood,
            "aic": float(aic),
            "r2": None,
            "iterations": iteration,
            "converged": converged,
        }

    if family == "poisson":
        if np.any(y < 0) or np.any(np.floor(y) != y):
            raise ValueError("Poisson GLR requires non-negative integer count values.")
        beta = np.zeros(p + 1)
        converged = False
        for iteration in range(1, max_iter + 1):
            eta = np.clip(X @ beta, -30.0, 30.0)
            mu = np.maximum(np.exp(eta), 1e-9)
            z = eta + (y - mu) / mu
            xtw = X.T * mu
            new_beta = np.linalg.pinv(xtw @ X) @ xtw @ z
            if np.max(np.abs(new_beta - beta)) < tol:
                beta = new_beta
                converged = True
                break
            beta = new_beta
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = np.maximum(np.exp(eta), 1e-9)
        cov = np.linalg.pinv((X.T * mu) @ X)
        se = np.sqrt(np.maximum(0.0, np.diagonal(cov)))
        z_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        p_values = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(z_stats) / math.sqrt(2.0))))
        log_factorial = np.array([math.lgamma(float(value) + 1.0) for value in y], dtype=float)
        log_likelihood = float(np.sum(y * np.log(mu) - mu - log_factorial))
        aic = -2.0 * log_likelihood + 2.0 * (p + 1)
        residuals = y - mu
        return {
            "family": "poisson",
            "coefficients": beta,
            "std_errors": se,
            "z_statistics": z_stats,
            "p_values": p_values,
            "fitted": mu,
            "residuals": residuals,
            "log_likelihood": log_likelihood,
            "aic": float(aic),
            "r2": None,
            "iterations": iteration,
            "converged": converged,
        }

    raise ValueError("Unsupported GLR family. Use gaussian, logistic, or poisson.")


def calculate_quantile_regression(
    y: np.ndarray,
    X_data: np.ndarray,
    tau: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Linear quantile regression via iteratively reweighted least squares
    (a Schlossmacher-style majorize-minimize scheme on the pinball loss):
    each iteration re-weights observations by tau/|residual| (residual >= 0)
    or (1-tau)/|residual| (residual < 0) and re-solves weighted least
    squares, converging to the tau-quantile fit. No asymptotic standard
    errors are produced - only coefficients, fitted values, and a pseudo-R2
    (Koenker & Machado 1999 R1) comparing model pinball loss to an
    intercept-only (unconditional quantile) model."""
    n = len(y)
    p = X_data.shape[1]
    if n <= p + 1:
        raise ValueError("Quantile regression requires more observations than model parameters.")
    if not (0.0 < tau < 1.0):
        raise ValueError("Quantile (tau) must be strictly between 0 and 1.")

    X = np.column_stack((np.ones(n), X_data))
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    eps = 1e-6
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        residuals = y - X @ beta
        abs_resid = np.maximum(np.abs(residuals), eps)
        asymmetric = np.where(residuals >= 0, tau, 1.0 - tau)
        weights = asymmetric / abs_resid
        xtw = X.T * weights
        new_beta = np.linalg.pinv(xtw @ X) @ xtw @ y
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            converged = True
            break
        beta = new_beta

    fitted = X @ beta
    residuals = y - fitted

    def _pinball_loss(resid: np.ndarray) -> float:
        return float(np.sum(np.where(resid >= 0, tau * resid, (tau - 1.0) * resid)))

    null_fit = np.quantile(y, tau)
    null_loss = _pinball_loss(y - null_fit)
    model_loss = _pinball_loss(residuals)
    pseudo_r2 = (1.0 - model_loss / null_loss) if null_loss > 0 else None

    return {
        "tau": tau,
        "coefficients": beta,
        "fitted": fitted,
        "residuals": residuals,
        "pseudo_r2": pseudo_r2,
        "iterations": iteration,
        "converged": converged,
    }


def calculate_gw_summary_stats(
    values: np.ndarray,
    coords: np.ndarray,
    bandwidth: float,
    kernel_type: str = "adaptive_bisquare",
) -> dict:
    """Geographically weighted local mean/std/skew of one field, using the
    same kernel families as calculate_gwr (fixed_gaussian, fixed_bisquare,
    adaptive_bisquare) so bandwidth intuition carries over from GWR/MGWR.
    Also reports each point's Kish effective sample size (w_sum^2 /
    sum(w^2)) - a rough count of how many observations are really informing
    that point's local statistics, useful for spotting sparse-data areas
    where the local estimate should be trusted less."""
    n = len(values)
    diff = coords[:, None, :] - coords[None, :, :]
    dists_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    local_mean = np.full(n, np.nan)
    local_std = np.full(n, np.nan)
    local_skew = np.full(n, np.nan)
    effective_n = np.zeros(n)

    for i in range(n):
        dists = dists_matrix[i]
        if kernel_type == "fixed_gaussian":
            w = np.exp(-0.5 * (dists / bandwidth) ** 2)
        elif kernel_type == "fixed_bisquare":
            w = np.zeros(n)
            mask = dists < bandwidth
            w[mask] = (1.0 - (dists[mask] / bandwidth) ** 2) ** 2
        elif kernel_type == "adaptive_bisquare":
            k = int(bandwidth)
            d_k = np.sort(dists)[min(k - 1, n - 1)]
            w = np.zeros(n)
            if d_k > 0:
                mask = dists < d_k
                w[mask] = (1.0 - (dists[mask] / d_k) ** 2) ** 2
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

        w_sum = float(np.sum(w))
        if w_sum <= 0:
            continue
        weights_norm = w / w_sum
        mean_i = float(np.sum(weights_norm * values))
        var_i = float(np.sum(weights_norm * (values - mean_i) ** 2))
        std_i = math.sqrt(max(var_i, 0.0))
        skew_i = (float(np.sum(weights_norm * (values - mean_i) ** 3)) / (std_i ** 3)) if std_i > 1e-9 else 0.0
        local_mean[i] = mean_i
        local_std[i] = std_i
        local_skew[i] = skew_i
        w_sq_sum = float(np.sum(w ** 2))
        effective_n[i] = (w_sum ** 2 / w_sq_sum) if w_sq_sum > 0 else 0.0

    return {"local_mean": local_mean, "local_std": local_std, "local_skew": local_skew, "effective_n": effective_n}


def calculate_bivariate_local_moran(
    x: np.ndarray,
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    permutations: int = 999,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Calculates Bivariate Local Moran's I (LISA) cluster and outlier diagnostics using conditional permutation.

    Returns:
        A tuple of:
          - I_values: NumPy array of Bivariate Moran's I indices (floats)
          - z_scores: NumPy array of z-scores (floats)
          - p_values: NumPy array of pseudo p-values (floats)
          - quadrants: List of strings ('HH', 'LL', 'HL', 'LH', 'Not Significant')
    """
    n = len(x)
    I_values = np.zeros(n)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    quadrants = ["Not Significant"] * n

    if n <= 2:
        return I_values, z_scores, p_values, quadrants

    x_mean = np.mean(x)
    x_std = np.std(x)
    y_mean = np.mean(y)
    y_std = np.std(y)

    if x_std == 0 or y_std == 0:
        return I_values, z_scores, p_values, quadrants

    zx = (x - x_mean) / x_std
    zy = (y - y_mean) / y_std

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    # Compute observed local Moran's I for each location
    spatial_lags = np.zeros(n)
    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])

        valid_neigh_indices = []
        valid_w = []
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx:
                valid_neigh_indices.append(id_to_idx[nid])
                valid_w.append(f_weights[j])

        w_sum = sum(valid_w)
        if w_sum == 0:
            continue

        w_arr = np.array(valid_w) / w_sum
        lag = np.sum(w_arr * zy[valid_neigh_indices])
        spatial_lags[idx] = lag
        I_values[idx] = zx[idx] * lag

    class _LCG:
        __slots__ = ("_s",)
        _A = 6364136223846793005
        _C = 1442695040888963407
        _M = (1 << 64) - 1
        def __init__(self, s):
            self._s = (int(s) * 2 + 0x9E3779B97F4A7C15) & self._M
        def uniform(self, lo, hi):
            self._s = (self._A * self._s + self._C) & self._M
            frac = (self._s >> 11) / float(1 << 53)
            return lo + (hi - lo) * frac
        def shuffle(self, arr):
            for i in range(len(arr) - 1, 0, -1):
                j = int(self.uniform(0, i + 1))
                arr[i], arr[j] = arr[j], arr[i]

    rng = _LCG(seed)

    # Perform conditional permutation testing
    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])

        valid_neigh_indices = []
        valid_w = []
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx:
                valid_neigh_indices.append(id_to_idx[nid])
                valid_w.append(f_weights[j])

        w_sum = sum(valid_w)
        if w_sum == 0:
            continue

        w_arr = np.array(valid_w) / w_sum
        observed_I = I_values[idx]

        other_indices = [i for i in range(n) if i != idx]
        perm_I_vals = np.zeros(permutations)
        num_neighbors = len(valid_neigh_indices)

        for p in range(permutations):
            shuffled_others = list(other_indices)
            rng.shuffle(shuffled_others)
            perm_neighbor_indices = shuffled_others[:num_neighbors]

            perm_lag = np.sum(w_arr * zy[perm_neighbor_indices])
            perm_I_vals[p] = zx[idx] * perm_lag

        extreme_count = np.sum(np.abs(perm_I_vals) >= np.abs(observed_I))
        p_val = (extreme_count + 1) / (permutations + 1)
        p_values[idx] = p_val

        mean_perm = np.mean(perm_I_vals)
        std_perm = np.std(perm_I_vals)
        if std_perm > 0:
            z_scores[idx] = (observed_I - mean_perm) / std_perm
        else:
            z_scores[idx] = 0.0

        if p_val < 0.05:
            high_x = zx[idx] > 0
            high_lag = spatial_lags[idx] > 0

            if high_x and high_lag:
                quadrants[idx] = "HH"
            elif not high_x and not high_lag:
                quadrants[idx] = "LL"
            elif high_x and not high_lag:
                quadrants[idx] = "HL"
            elif not high_x and high_lag:
                quadrants[idx] = "LH"

    return I_values, z_scores, p_values, quadrants
