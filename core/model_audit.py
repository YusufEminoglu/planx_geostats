# -*- coding: utf-8 -*-
"""Model-audit scoring helpers for PlanX GeoStats Lab."""
from __future__ import annotations


def assign_model_scores(comparisons: list[dict]) -> None:
    """Mutate usable comparison dictionaries with score and rank values."""
    usable = [item for item in comparisons if item.get("usable")]
    if not usable:
        return
    rmse_values = [item["fit"]["rmse"] for item in usable if item["fit"].get("rmse") is not None]
    mae_values = [item["fit"]["mae"] for item in usable if item["fit"].get("mae") is not None]
    max_rmse = max(max(rmse_values) if rmse_values else 1.0, 1.0e-9)
    max_mae = max(max(mae_values) if mae_values else 1.0, 1.0e-9)

    for item in usable:
        fit = item["fit"]
        residual = item.get("residual_spatial") or {}
        rmse_component = (fit["rmse"] / max_rmse) if fit.get("rmse") is not None else 1.0
        mae_component = (fit["mae"] / max_mae) if fit.get("mae") is not None else 1.0
        residual_penalty = residual_pattern_penalty(residual)
        coverage_penalty = max(0.0, 1.0 - float(item.get("coverage", 1.0)))
        item["score"] = float(0.45 * rmse_component + 0.25 * mae_component + residual_penalty + 0.15 * coverage_penalty)

    for rank, item in enumerate(sorted(usable, key=lambda row: row["score"]), start=1):
        item["rank"] = rank


def residual_pattern_penalty(residual_summary: dict) -> float:
    """Return a model-audit penalty for residual spatial-pattern diagnostics."""
    if not residual_summary.get("available"):
        return 0.25
    p_value = residual_summary.get("p_value")
    if p_value is not None and p_value < 0.05:
        return 0.35
    return 0.0


def model_recommendation(comparisons: list[dict]) -> str:
    """Return a concise recommendation after scores/ranks are assigned."""
    usable = [item for item in comparisons if item.get("usable")]
    if not usable:
        return "No usable model outputs were available for comparison."
    if any("score" not in item or "rank" not in item for item in usable):
        assign_model_scores(usable)
    clean = [
        item for item in usable
        if item.get("residual_spatial", {}).get("available")
        and item.get("residual_spatial", {}).get("p_value") is not None
        and item["residual_spatial"]["p_value"] >= 0.05
    ]
    if clean:
        recommended = min(clean, key=lambda item: item["score"])
        return (
            f"{recommended['layer_name']} has the strongest audit score among models without a strong global residual spatial pattern. "
            "Review assumptions and coefficient meaning before selecting it."
        )
    best = min(usable, key=lambda item: item["score"])
    return (
        f"{best['layer_name']} has the strongest audit score, but every comparable model either retains residual spatial structure "
        "or lacks a residual diagnostic. Treat the comparison as unresolved and inspect residual maps."
    )
