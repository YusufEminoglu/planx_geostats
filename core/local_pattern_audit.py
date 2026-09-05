# -*- coding: utf-8 -*-
"""Local-pattern output summaries for hot spot and LISA tools."""
from __future__ import annotations

from collections import Counter

import numpy as np


MULTIPLE_TESTING_OPTIONS = (
    ("None (raw p-values)", "none"),
    ("Benjamini-Hochberg FDR", "fdr_bh"),
    ("Bonferroni family-wise error", "bonferroni"),
)


GI_CONF_LABELS = {
    -3: "99% cold spot",
    -2: "95% cold spot",
    -1: "90% cold spot",
    0: "not significant",
    1: "90% hot spot",
    2: "95% hot spot",
    3: "99% hot spot",
}

LISA_LABELS = {
    "HH": "high-high cluster",
    "LL": "low-low cluster",
    "HL": "high-low outlier",
    "LH": "low-high outlier",
    "Not Significant": "not significant",
}


def adjust_p_values(p_values, method: str = "fdr_bh") -> np.ndarray:
    """Return multiplicity-adjusted p-values without optional dependencies."""
    raw = np.clip(np.asarray(p_values, dtype=float), 0.0, 1.0)
    if method == "none" or raw.size == 0:
        return raw.copy()
    if method == "bonferroni":
        return np.minimum(1.0, raw * raw.size)
    if method != "fdr_bh":
        raise ValueError(f"Unknown multiple-testing correction: {method}")

    order = np.argsort(raw, kind="stable")
    ranked = raw[order]
    adjusted_ranked = ranked * raw.size / np.arange(1, raw.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def corrected_lisa_classes(quadrants, adjusted_p, alpha: float = 0.05) -> list[str]:
    """Downgrade raw LISA classes that do not survive multiplicity control."""
    return [
        str(klass) if float(p_value) < alpha else "Not Significant"
        for klass, p_value in zip(quadrants, adjusted_p)
    ]


def corrected_gi_bins(z_scores, adjusted_p) -> np.ndarray:
    """Rebuild signed Gi* confidence bins from adjusted p-values."""
    bins = np.zeros(len(adjusted_p), dtype=int)
    for index, (z_score, p_value) in enumerate(zip(z_scores, adjusted_p)):
        level = 3 if p_value < 0.01 else 2 if p_value < 0.05 else 1 if p_value < 0.10 else 0
        bins[index] = level if float(z_score) >= 0 else -level
    return bins


def getis_ord_class_summary(confidence_bins) -> dict:
    """Summarize Getis-Ord Gi* confidence-bin output."""
    counts = Counter(int(value) for value in confidence_bins if value is not None)
    hot_count = sum(count for klass, count in counts.items() if klass > 0)
    cold_count = sum(count for klass, count in counts.items() if klass < 0)
    significant_count = hot_count + cold_count
    dominant = _dominant_label(counts, GI_CONF_LABELS)
    return {
        "counts": dict(sorted(counts.items())),
        "hot_count": int(hot_count),
        "cold_count": int(cold_count),
        "significant_count": int(significant_count),
        "dominant_label": dominant,
        "message": _gi_message(hot_count, cold_count, significant_count, dominant),
    }


def local_moran_class_summary(quadrants) -> dict:
    """Summarize Local Moran cluster/outlier quadrant output."""
    normalized = [str(value) if value else "Not Significant" for value in quadrants]
    counts = Counter(normalized)
    cluster_count = counts.get("HH", 0) + counts.get("LL", 0)
    outlier_count = counts.get("HL", 0) + counts.get("LH", 0)
    dominant = _dominant_label(counts, LISA_LABELS)
    return {
        "counts": dict(sorted(counts.items())),
        "cluster_count": int(cluster_count),
        "outlier_count": int(outlier_count),
        "significant_count": int(cluster_count + outlier_count),
        "dominant_label": dominant,
        "message": _lisa_message(cluster_count, outlier_count, dominant),
    }


def _dominant_label(counts: Counter, labels: dict) -> str:
    if not counts:
        return "no classified features"
    klass, _ = max(counts.items(), key=lambda item: (item[1], str(item[0])))
    return labels.get(klass, str(klass))


def _gi_message(hot_count: int, cold_count: int, significant_count: int, dominant: str) -> str:
    if significant_count == 0:
        return "No statistically significant hot or cold spot classes were produced."
    return (
        f"Gi* classified {hot_count} hot spot feature(s) and {cold_count} cold spot feature(s); "
        f"the dominant class is {dominant}."
    )


def _lisa_message(cluster_count: int, outlier_count: int, dominant: str) -> str:
    if cluster_count + outlier_count == 0:
        return "No statistically significant Local Moran cluster or outlier classes were produced."
    return (
        f"Local Moran classified {cluster_count} cluster feature(s) and {outlier_count} outlier feature(s); "
        f"the dominant class is {dominant}."
    )
