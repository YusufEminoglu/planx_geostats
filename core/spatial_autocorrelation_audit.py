# -*- coding: utf-8 -*-
"""Interpretation helpers for global spatial autocorrelation reports."""
from __future__ import annotations


def global_moran_interpretation(z_score: float, p_value: float, neighborhood_summary: dict | None = None) -> dict:
    """Return report-ready interpretation for a Global Moran's I result."""
    is_significant = float(p_value) < 0.05
    if is_significant and z_score > 0:
        pattern = "Clustered"
        color = "#e31a1c"
        description = (
            f"Given the z-score of {z_score:.2f}, there is a less than 5% likelihood "
            "that this clustered pattern could be the result of random chance."
        )
    elif is_significant and z_score < 0:
        pattern = "Dispersed"
        color = "#1f78b4"
        description = (
            f"Given the z-score of {z_score:.2f}, there is a less than 5% likelihood "
            "that this dispersed pattern could be the result of random chance."
        )
    else:
        pattern = "Random"
        color = "#718096"
        description = (
            f"Given the z-score of {z_score:.2f}, the spatial pattern appears compatible "
            "with random spatial arrangement for this neighborhood definition."
        )

    return {
        "pattern": pattern,
        "color": color,
        "description": description,
        "confidence": evidence_strength(p_value),
        "next_action": global_moran_next_action(is_significant, neighborhood_summary or {}),
    }


def evidence_strength(p_value: float) -> str:
    """Return a compact evidence-strength label for report copy."""
    p_value = float(p_value)
    if p_value < 0.01:
        return "very strong"
    if p_value < 0.05:
        return "strong"
    if p_value < 0.10:
        return "suggestive but not conventionally significant"
    return "weak"


def global_moran_next_action(is_significant: bool, neighborhood_summary: dict) -> str:
    """Return the most important next analyst action for the neighborhood graph."""
    if int(neighborhood_summary.get("isolated", 0) or 0) > 0:
        return "Increase the distance band or choose KNN weights before using this result for planning decisions."
    if bool(neighborhood_summary.get("all_connected", False)):
        return "Try a smaller threshold or a data-driven distance band to avoid masking local structure."
    if is_significant:
        return "Follow up with Local Moran's I or Gi* to locate the neighborhoods driving this global pattern."
    return "Review scale, zoning geography, and candidate distance bands before concluding that the process is spatially random."
