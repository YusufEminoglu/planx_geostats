# -*- coding: utf-8 -*-
"""Spatial statistics engines for calculations."""
from __future__ import annotations

import logging
import math
import numpy as np

logger = logging.getLogger("PlanX-GeoStats")

# Try importing PySAL modules
HAS_PYQ = False
try:
    from esda.g import Gi_Local
    from esda.moran import Moran_Local
    import libpysal
    HAS_PYQ = True
except ImportError:
    pass


def calculate_getis_ord(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    star: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the Getis-Ord Gi or Gi* statistics."""
    n = len(y)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    conf_bins = np.zeros(n, dtype=int)

    if n <= 1:
        return z_scores, p_values, conf_bins

    y_mean = np.mean(y)
    y_std = np.std(y)

    if y_std == 0:
        logger.warning("Standard deviation of the target field is zero. Gi* cannot be calculated.")
        return z_scores, p_values, conf_bins

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        valid_neigh_indices = [id_to_idx[nid] for nid in f_neighs if nid in id_to_idx]

        if star:
            if idx not in valid_neigh_indices:
                valid_neigh_indices.append(idx)

        num_neighbors = len(valid_neigh_indices)
        if num_neighbors == 0:
            continue

        w_row = np.ones(num_neighbors) / num_neighbors
        y_neigh = y[valid_neigh_indices]
        sum_w_x = np.sum(w_row * y_neigh)
        sum_w = np.sum(w_row)
        sum_w2 = np.sum(w_row ** 2)

        numerator = sum_w_x - y_mean * sum_w
        denom_term = (n * sum_w2 - (sum_w ** 2)) / (n - 1)
        if denom_term < 0:
            denom_term = 0.0
        denominator = y_std * math.sqrt(denom_term)

        if denominator > 0:
            z = numerator / denominator
            z_scores[idx] = z
            p = 1.0 - math.erf(abs(z) / math.sqrt(2.0))
            p_values[idx] = p

            if p < 0.01:
                conf_bins[idx] = 3 if z > 0 else -3
            elif p < 0.05:
                conf_bins[idx] = 2 if z > 0 else -2
            elif p < 0.10:
                conf_bins[idx] = 1 if z > 0 else -1
            else:
                conf_bins[idx] = 0

    return z_scores, p_values, conf_bins


def calculate_mean_center(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: np.ndarray | None = None
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
    weights: np.ndarray | None = None
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
    weights: np.ndarray | None = None,
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
    n = len(y)
    I_values = np.zeros(n)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    quadrants = ["Not Significant"] * n

    if n <= 2:
        return I_values, z_scores, p_values, quadrants

    y_mean = np.mean(y)
    z = y - y_mean
    m2 = np.sum(z ** 2) / n

    if m2 == 0:
        return I_values, z_scores, p_values, quadrants

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
            # Normal approximation for t-dist (very accurate for large df)
            p_vals[j] = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stats[j]) / math.sqrt(2.0))))
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
        except Exception:
            pass

    # --- DIAGNOSTIC 3: Moran's I on Residuals ---
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    spatial_lag_e = np.zeros(n)
    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])
        lag_sum = 0.0
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx:
                lag_sum += f_weights[j] * residuals[id_to_idx[nid]]
        spatial_lag_e[idx] = lag_sum

    if ss_res > 0:
        moran_i = np.sum(residuals * spatial_lag_e) / ss_res
    else:
        moran_i = 0.0

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
        "residuals": residuals,
        "std_residuals": std_residuals,
        "jarque_bera": (jb_stat, jb_p),
        "breusch_pagan": (bp_stat, bp_p),
        "residuals_moran": moran_i,
        "variable_names": ["Intercept"] + x_names
    }
