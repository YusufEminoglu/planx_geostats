# -*- coding: utf-8 -*-
"""Shared diagnostics for PlanX GeoStats analytical reports."""
from __future__ import annotations

import html
import math
from typing import Iterable, Optional

import numpy as np


def numeric_quality_summary(total_features: int, valid_values: dict, values: Iterable[float]) -> dict:
    arr = np.array(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    skipped = max(0, int(total_features) - int(len(valid_values)))
    summary = {
        "total_features": int(total_features),
        "valid_numeric": int(len(finite)),
        "skipped": skipped,
        "minimum": None,
        "maximum": None,
        "mean": None,
        "std": None,
        "is_constant": False,
        "has_non_finite": int(len(arr)) != int(len(finite)),
    }
    if len(finite) > 0:
        summary.update({
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "is_constant": bool(np.std(finite) == 0.0),
        })
    return summary


def neighbor_summary(neighbors: dict, valid_ids: Iterable[int]) -> dict:
    valid_set = set(valid_ids)
    counts = []
    for fid in valid_set:
        counts.append(len([nid for nid in neighbors.get(fid, []) if nid in valid_set]))

    if not counts:
        return {
            "minimum": 0,
            "median": 0.0,
            "maximum": 0,
            "mean": 0.0,
            "isolated": 0,
            "all_connected": False,
            "density_label": "No valid neighborhood graph",
        }

    arr = np.array(counts, dtype=float)
    n = len(counts)
    maximum_possible = max(0, n - 1)
    isolated = int(np.sum(arr == 0))
    all_connected = bool(maximum_possible > 0 and np.all(arr >= maximum_possible))
    mean_neighbors = float(np.mean(arr))
    if isolated > 0:
        density_label = "Sparse with isolated observations"
    elif maximum_possible > 0 and mean_neighbors >= 0.8 * maximum_possible:
        density_label = "Very dense neighborhood graph"
    elif mean_neighbors <= 1.0:
        density_label = "Sparse neighborhood graph"
    else:
        density_label = "Usable neighborhood graph"

    return {
        "minimum": int(np.min(arr)),
        "median": float(np.median(arr)),
        "maximum": int(np.max(arr)),
        "mean": mean_neighbors,
        "isolated": isolated,
        "all_connected": all_connected,
        "density_label": density_label,
    }


def filter_weights_to_valid_ids(neighbors: dict, valid_ids: Iterable[int]) -> tuple[dict[int, list[int]], dict[int, list[float]], list[int]]:
    """Filter a neighbor graph to complete records and rebuild row-standardized weights."""
    id_order = [int(fid) for fid in valid_ids]
    valid_set = set(id_order)
    filtered_neighbors: dict[int, list[int]] = {}
    filtered_weights: dict[int, list[float]] = {}

    for fid in id_order:
        row_neighbors = [int(nid) for nid in neighbors.get(fid, []) if int(nid) in valid_set]
        filtered_neighbors[fid] = row_neighbors
        if row_neighbors:
            filtered_weights[fid] = [1.0 / len(row_neighbors)] * len(row_neighbors)
        else:
            filtered_weights[fid] = []

    return filtered_neighbors, filtered_weights, id_order


def residual_spatial_autocorrelation_summary(
    residuals: Iterable[float],
    neighbors: dict,
    weights: dict,
    id_order: Iterable[int],
) -> dict:
    """Calculate a compact Global Moran's I diagnostic for model residuals."""
    ids = [int(fid) for fid in id_order]
    arr = np.array(list(residuals), dtype=float)
    finite_mask = np.isfinite(arr)
    if len(ids) != len(arr):
        return _empty_residual_spatial_summary("Residual count does not match the feature-id order.")
    if len(arr) < 4:
        return _empty_residual_spatial_summary("At least 4 complete residuals are required.")
    if not np.all(finite_mask):
        ids = [fid for fid, ok in zip(ids, finite_mask) if ok]
        arr = arr[finite_mask]
    if len(arr) < 4:
        return _empty_residual_spatial_summary("At least 4 finite residuals are required.")

    filtered_neighbors, filtered_weights, ids = filter_weights_to_valid_ids(neighbors, ids)
    n_summary = neighbor_summary(filtered_neighbors, ids)
    id_to_idx = {fid: idx for idx, fid in enumerate(ids)}
    centered = arr - np.mean(arr)
    sum_z2 = float(np.sum(centered ** 2))
    if sum_z2 <= 0.0:
        return {
            **_empty_residual_spatial_summary("Residuals are constant; residual Moran's I is not informative."),
            "available": True,
            "moran_i": 0.0,
            "expected_i": -1.0 / (len(arr) - 1),
            "p_value": 1.0,
            "z_score": 0.0,
            "neighbor_summary": n_summary,
            "status": "No residual contrast",
        }

    s0 = 0.0
    row_sums = np.zeros(len(ids))
    col_sums = np.zeros(len(ids))
    numerator = 0.0
    for i, fid in enumerate(ids):
        for nid, weight in zip(filtered_neighbors.get(fid, []), filtered_weights.get(fid, [])):
            if nid in id_to_idx:
                j = id_to_idx[nid]
                w = float(weight)
                s0 += w
                row_sums[i] += w
                col_sums[j] += w
                numerator += w * centered[i] * centered[j]

    if s0 <= 0.0:
        return {
            **_empty_residual_spatial_summary("No valid residual-neighbor links were available."),
            "neighbor_summary": n_summary,
            "status": "No valid residual-neighbor links",
        }

    n = len(ids)
    moran_i = (n / s0) * (numerator / sum_z2)
    expected_i = -1.0 / (n - 1)
    sum_z4 = float(np.sum(centered ** 4))
    kurtosis_term = (n * sum_z4) / (sum_z2 ** 2) if sum_z2 > 0 else 0.0

    s1 = 0.0
    for fid in ids:
        for nid, w_ij in zip(filtered_neighbors.get(fid, []), filtered_weights.get(fid, [])):
            if nid in id_to_idx:
                w_ji = 0.0
                reverse_neighbors = filtered_neighbors.get(nid, [])
                reverse_weights = filtered_weights.get(nid, [])
                if fid in reverse_neighbors:
                    w_ji = reverse_weights[reverse_neighbors.index(fid)]
                s1 += (float(w_ij) + float(w_ji)) ** 2
    s1 *= 0.5
    s2 = float(np.sum((row_sums + col_sums) ** 2))
    numerator_variance = (
        n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2)
        - kurtosis_term * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * s0 ** 2)
    )
    denominator_variance = (n - 1) * (n - 2) * (n - 3) * s0 ** 2
    variance = numerator_variance / denominator_variance - (expected_i ** 2) if denominator_variance > 0 else 0.0
    if variance > 0:
        z_score = (moran_i - expected_i) / math.sqrt(variance)
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
    else:
        variance = 0.0
        z_score = 0.0
        p_value = 1.0

    if p_value < 0.05 and moran_i > expected_i:
        status = "Residual clustering remains"
    elif p_value < 0.05 and moran_i < expected_i:
        status = "Residual dispersion remains"
    else:
        status = "No strong residual spatial pattern"

    return {
        "available": True,
        "moran_i": float(moran_i),
        "expected_i": float(expected_i),
        "variance": float(variance),
        "z_score": float(z_score),
        "p_value": float(max(0.0, min(1.0, p_value))),
        "neighbor_summary": n_summary,
        "status": status,
        "message": "",
    }


def _empty_residual_spatial_summary(message: str) -> dict:
    return {
        "available": False,
        "moran_i": None,
        "expected_i": None,
        "variance": None,
        "z_score": None,
        "p_value": None,
        "neighbor_summary": None,
        "status": "Not available",
        "message": message,
    }


def push_residual_spatial_diagnostics(feedback, summary: dict) -> None:
    if not summary.get("available"):
        feedback.pushWarning("Residual spatial autocorrelation diagnostic was not available: " + summary.get("message", "unknown reason"))
        return
    feedback.pushInfo(
        "Residual spatial autocorrelation: "
        f"Moran's I={format_number(summary['moran_i'], 6)}, "
        f"z={format_number(summary['z_score'], 4)}, "
        f"p={format_number(summary['p_value'], 6)}, "
        f"status={summary['status']}."
    )
    if summary.get("p_value") is not None and summary["p_value"] < 0.05:
        feedback.pushWarning("Residuals retain a statistically notable spatial pattern. Review model specification and missing spatial processes.")


def crs_unit_warning(source) -> Optional[str]:
    try:
        crs = source.sourceCrs()
    except Exception:
        return None
    try:
        if crs.isGeographic():
            return (
                "The input CRS appears to be geographic. Distance-based statistics use map units, "
                "so project the layer to an appropriate local projected CRS before relying on distance bands."
            )
    except Exception:
        return None
    return None


def push_diagnostics(feedback, numeric_summary: dict, neighborhood_summary: Optional[dict] = None, crs_warning: Optional[str] = None) -> None:
    feedback.pushInfo(
        "Input diagnostics: "
        f"{numeric_summary['valid_numeric']} valid numeric feature(s), "
        f"{numeric_summary['skipped']} skipped/null/non-numeric feature(s), "
        f"{numeric_summary['total_features']} total feature(s)."
    )
    if numeric_summary.get("is_constant"):
        feedback.pushWarning("The analysis field is constant. Significance statistics may be uninformative.")
    if numeric_summary.get("has_non_finite"):
        feedback.pushWarning("Non-finite numeric values were detected and excluded from diagnostics.")
    if crs_warning:
        feedback.pushWarning(crs_warning)
    if neighborhood_summary:
        feedback.pushInfo(
            "Neighborhood diagnostics: "
            f"min={neighborhood_summary['minimum']}, "
            f"median={neighborhood_summary['median']:.2f}, "
            f"max={neighborhood_summary['maximum']}, "
            f"isolated={neighborhood_summary['isolated']}."
        )
        if neighborhood_summary["isolated"] > 0:
            feedback.pushWarning("Some observations have no valid neighbors. Review the spatial relationship settings.")
        if neighborhood_summary["all_connected"]:
            feedback.pushWarning("Every feature is connected to every other feature. The neighborhood may be too broad.")


def format_number(value, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        if not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.{precision}f}"
    except Exception:
        return "n/a"


def diagnostics_html(numeric_summary: dict, neighborhood_summary: Optional[dict], crs_warning: Optional[str]) -> str:
    rows = [
        ("Total features", str(numeric_summary["total_features"])),
        ("Valid numeric features", str(numeric_summary["valid_numeric"])),
        ("Skipped/null/non-numeric features", str(numeric_summary["skipped"])),
        ("Minimum value", format_number(numeric_summary["minimum"])),
        ("Maximum value", format_number(numeric_summary["maximum"])),
        ("Mean value", format_number(numeric_summary["mean"])),
        ("Standard deviation", format_number(numeric_summary["std"])),
    ]
    if neighborhood_summary:
        rows.extend([
            ("Minimum neighbors", str(neighborhood_summary["minimum"])),
            ("Median neighbors", f"{neighborhood_summary['median']:.2f}"),
            ("Maximum neighbors", str(neighborhood_summary["maximum"])),
            ("Isolated observations", str(neighborhood_summary["isolated"])),
            ("Neighborhood density", neighborhood_summary["density_label"]),
        ])
    if crs_warning:
        rows.append(("CRS warning", crs_warning))

    body = "".join(
        "<tr>"
        f"<td class=\"metric-name\">{html.escape(label)}</td>"
        f"<td class=\"metric-val\">{html.escape(value)}</td>"
        "</tr>"
        for label, value in rows
    )
    return (
        "<section><h2>Input and Neighborhood Diagnostics</h2>"
        "<table><thead><tr><th>Diagnostic</th><th>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table></section>"
    )


def caveats_html(method_name: str, neighborhood_summary: Optional[dict], numeric_summary: dict) -> str:
    caveats = [
        f"{method_name} evaluates spatial pattern under the selected spatial relationship; changing the distance band or K value can change the conclusion.",
        "Interpret statistical significance together with planning context, data collection process, and known spatial boundaries.",
    ]
    if numeric_summary.get("is_constant"):
        caveats.append("The analysis field is constant, so pattern statistics cannot provide meaningful contrast.")
    if neighborhood_summary:
        if neighborhood_summary["isolated"] > 0:
            caveats.append("Isolated observations reduce local support and may indicate that the threshold distance is too small.")
        if neighborhood_summary["all_connected"]:
            caveats.append("A fully connected graph can hide local structure; consider a smaller distance band or a data-driven threshold.")
    items = "".join(f"<li>{html.escape(item)}</li>" for item in caveats)
    return f"<section><h2>Assumptions and Caveats</h2><ul>{items}</ul></section>"


def regression_quality_summary(y: np.ndarray, x_data: np.ndarray, x_names: list[str], total_features: int) -> dict:
    n = int(len(y))
    p = int(x_data.shape[1]) if x_data.ndim == 2 else 0
    skipped = max(0, int(total_features) - n)
    near_constant = []
    for idx, name in enumerate(x_names):
        column = x_data[:, idx]
        if float(np.std(column)) <= 1e-9:
            near_constant.append(name)

    corr_warnings = []
    max_abs_corr = 0.0
    if p > 1 and n > 2:
        corr = np.corrcoef(x_data, rowvar=False)
        for i in range(p):
            for j in range(i + 1, p):
                value = corr[i, j]
                if np.isfinite(value):
                    max_abs_corr = max(max_abs_corr, abs(float(value)))
                    if abs(value) >= 0.85:
                        corr_warnings.append((x_names[i], x_names[j], float(value)))

    condition_number = None
    if p > 0 and n > p:
        design = np.column_stack((np.ones(n), x_data))
        try:
            condition_number = float(np.linalg.cond(design))
        except Exception:
            condition_number = None

    risks = []
    if n <= (p + 1) * 5:
        risks.append("Sample size is small relative to the number of model parameters.")
    if near_constant:
        risks.append("One or more predictors are constant or nearly constant.")
    if corr_warnings:
        risks.append("High pairwise predictor correlation suggests possible multicollinearity.")
    if condition_number is not None and condition_number >= 30:
        risks.append("The design matrix condition number suggests unstable coefficient estimates.")

    return {
        "total_features": int(total_features),
        "used_records": n,
        "skipped_records": skipped,
        "predictor_count": p,
        "near_constant": near_constant,
        "high_correlations": corr_warnings,
        "max_abs_correlation": max_abs_corr,
        "condition_number": condition_number,
        "risks": risks,
    }


def regression_quality_html(summary: dict) -> str:
    corr_text = "None above 0.85"
    if summary["high_correlations"]:
        corr_text = "; ".join(
            f"{left} vs {right}: {corr:.3f}"
            for left, right, corr in summary["high_correlations"][:8]
        )
    rows = [
        ("Total input features", str(summary["total_features"])),
        ("Complete records used", str(summary["used_records"])),
        ("Skipped/incomplete records", str(summary["skipped_records"])),
        ("Predictor count", str(summary["predictor_count"])),
        ("Near-constant predictors", ", ".join(summary["near_constant"]) or "None"),
        ("Maximum absolute predictor correlation", format_number(summary["max_abs_correlation"], 3)),
        ("High-correlation pairs", corr_text),
        ("Condition number", format_number(summary["condition_number"], 3)),
    ]
    body = "".join(
        "<tr>"
        f"<td class=\"metric-name\">{html.escape(label)}</td>"
        f"<td>{html.escape(value)}</td>"
        "</tr>"
        for label, value in rows
    )
    risk_items = "".join(f"<li>{html.escape(risk)}</li>" for risk in summary["risks"])
    if not risk_items:
        risk_items = "<li>No major automatic model-quality warning was triggered.</li>"
    return (
        "<h2>Model Quality Checks</h2>"
        "<table><thead><tr><th>Check</th><th>Result</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f"<div class=\"note\"><strong>Analyst review:</strong><ul>{risk_items}</ul></div>"
    )


def residual_spatial_autocorrelation_html(summary: dict) -> str:
    if not summary.get("available"):
        return (
            "<h2>Residual Spatial Autocorrelation</h2>"
            "<div class=\"note\"><strong>Diagnostic unavailable:</strong> "
            f"{html.escape(summary.get('message', 'No diagnostic message was provided.'))}</div>"
        )
    n_summary = summary.get("neighbor_summary") or {}
    rows = [
        ("Residual Moran's I", format_number(summary["moran_i"], 6)),
        ("Expected I", format_number(summary["expected_i"], 6)),
        ("z-score", format_number(summary["z_score"], 4)),
        ("p-value", format_number(summary["p_value"], 6)),
        ("Status", summary["status"]),
        ("Minimum residual neighbors", str(n_summary.get("minimum", "n/a"))),
        ("Median residual neighbors", format_number(n_summary.get("median"), 2)),
        ("Maximum residual neighbors", str(n_summary.get("maximum", "n/a"))),
        ("Isolated residual observations", str(n_summary.get("isolated", "n/a"))),
    ]
    body = "".join(
        "<tr>"
        f"<td class=\"metric-name\">{html.escape(label)}</td>"
        f"<td>{html.escape(str(value))}</td>"
        "</tr>"
        for label, value in rows
    )
    if summary.get("p_value") is not None and summary["p_value"] < 0.05:
        interpretation = (
            "Residuals still show a spatial pattern after model fitting. This can indicate an omitted spatial process, "
            "a scale mismatch, misspecified neighborhoods, or explanatory variables that do not capture the geography of the outcome."
        )
    else:
        interpretation = (
            "The selected residual-neighborhood graph does not show a strong global residual pattern. "
            "Still inspect the residual map because local pockets can remain even when the global diagnostic is weak."
        )
    return (
        "<h2>Residual Spatial Autocorrelation</h2>"
        "<table><thead><tr><th>Diagnostic</th><th>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f"<div class=\"note\"><strong>Analyst interpretation:</strong> {html.escape(interpretation)}</div>"
    )
