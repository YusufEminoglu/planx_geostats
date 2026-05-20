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
