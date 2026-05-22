# -*- coding: utf-8 -*-
"""Sensitivity-audit helpers for Monte Carlo spatial autocorrelation reports."""
from __future__ import annotations


def sensitivity_verdict(results: dict, neighborhood_summary: dict | None = None) -> dict:
    """Return report-ready interpretation for an attribute-randomization test."""
    empirical_p = float(results.get("empirical_p", 1.0))
    observed_i = float(results.get("observed_i", 0.0))
    isolated = int((neighborhood_summary or {}).get("isolated", 0) or 0)
    density_label = (neighborhood_summary or {}).get("density_label", "")

    if empirical_p < 0.05:
        verdict = "ROBUST - Statistically Significant"
        color = "#2b9348"
        description = (
            f"The observed Moran's I ({observed_i:.6f}) is statistically significant "
            f"(empirical p = {empirical_p:.4f}). The spatial pattern is unlikely under random "
            "attribute reassignment for this neighborhood definition."
        )
    else:
        verdict = "SENSITIVE - Not Statistically Significant"
        color = "#d62728"
        description = (
            f"The observed Moran's I ({observed_i:.6f}) falls within the Monte Carlo reference "
            f"distribution (empirical p = {empirical_p:.4f}). The pattern is not robust enough "
            "to treat as planning evidence without additional checks."
        )

    cautions = []
    if isolated > 0:
        cautions.append("The neighborhood graph contains isolated observations; distance-band sensitivity is high.")
    if density_label == "Very dense neighborhood graph":
        cautions.append("The neighborhood graph is very dense; local differences may be over-smoothed.")
    if empirical_p >= 0.05:
        cautions.append("Run an alternative defensible distance band before interpreting the pattern.")
    if not cautions:
        cautions.append("No major automatic sensitivity warning was triggered.")

    return {
        "verdict": verdict,
        "color": color,
        "description": description,
        "next_action": sensitivity_next_action(empirical_p, isolated, density_label),
        "cautions": cautions,
    }


def sensitivity_next_action(empirical_p: float, isolated: int = 0, density_label: str = "") -> str:
    """Return a concise analyst action for the sensitivity-test outcome."""
    if isolated > 0:
        return "Increase the distance band or use a data-driven threshold before relying on this robustness result."
    if density_label == "Very dense neighborhood graph":
        return "Repeat the test with a smaller defensible distance band to confirm the result is not over-smoothed."
    if empirical_p < 0.05:
        return "Treat the spatial pattern as robust enough for follow-up local analysis, then map where the pattern is concentrated."
    return "Revisit variable choice, study area, and neighborhood definition before using this pattern as planning evidence."
