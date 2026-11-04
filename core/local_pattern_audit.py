# -*- coding: utf-8 -*-
"""Local-pattern output summaries for hot spot and LISA tools."""
from __future__ import annotations

from collections import Counter

import numpy as np


MULTIPLE_TESTING_OPTIONS = (
    ("None (raw p-values)", "none"),
    ("Benjamini-Hochberg false discovery rate (recommended)", "fdr_bh"),
    ("Bonferroni family-wise error", "bonferroni"),
)

# What a report says, in words, for each correction code (the code itself,
# e.g. "fdr_bh", stays in the *_mt field for scripts).
CORRECTION_WORDS = {
    "none": "no correction (raw p-values)",
    "fdr_bh": "Benjamini-Hochberg false discovery rate",
    "bonferroni": "Bonferroni family-wise error rate",
}

# Contract C2: the two classes for features where nothing was computed.
NO_DATA = "No data"
NO_NEIGHBOURS = "No neighbours"
MISSING_CLASSES = (NO_DATA, NO_NEIGHBOURS)

# Consensus classes of the Gi* stability analysis (they are not confidence levels).
STABLE_HOT = "Stable hot spot"
STABLE_COLD = "Stable cold spot"
NOT_STABLE = "Not stable"


def correction_words(method: str) -> str:
    """Plain-language name of a multiple-testing correction code."""
    return CORRECTION_WORDS.get(str(method), str(method))


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
    """Return multiplicity-adjusted p-values without optional dependencies.

    NaN p-values (features with no data or no neighbours, where no test was
    run) stay NaN and are not counted as tests: the family size is the number
    of finite p-values.  Before 3.13 one NaN made every FDR-adjusted value NaN.
    """
    all_raw = np.asarray(p_values, dtype=float)
    if method not in ("none", "bonferroni", "fdr_bh"):
        raise ValueError(f"Unknown multiple-testing correction: {method}")
    adjusted_all = np.full(all_raw.shape, np.nan)
    finite = np.isfinite(all_raw)
    raw = np.clip(all_raw[finite], 0.0, 1.0)
    if raw.size == 0:
        return adjusted_all
    if method == "none":
        adjusted = raw.copy()
    elif method == "bonferroni":
        adjusted = np.minimum(1.0, raw * raw.size)
    else:
        order = np.argsort(raw, kind="stable")
        ranked = raw[order]
        adjusted_ranked = ranked * raw.size / np.arange(1, raw.size + 1)
        adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
        adjusted = np.empty_like(adjusted_ranked)
        adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted_all[finite] = adjusted
    return adjusted_all


def corrected_lisa_classes(quadrants, adjusted_p, alpha: float = 0.05) -> list[str]:
    """Downgrade raw LISA classes that do not survive multiplicity control."""
    out = []
    for klass, p_value in zip(quadrants, adjusted_p):
        if str(klass) in MISSING_CLASSES:
            out.append(str(klass))
        elif p_value is not None and np.isfinite(float(p_value)) and float(p_value) < alpha:
            out.append(str(klass))
        else:
            out.append("Not Significant")
    return out


def corrected_gi_bins(z_scores, adjusted_p) -> np.ndarray:
    """Rebuild signed Gi* confidence bins from adjusted p-values."""
    bins = np.zeros(len(adjusted_p), dtype=int)
    for index, (z_score, p_value) in enumerate(zip(z_scores, adjusted_p)):
        if not (np.isfinite(float(p_value)) and np.isfinite(float(z_score))):
            continue  # no test was run: confidence 0 (contract C2)
        level = 3 if p_value < 0.01 else 2 if p_value < 0.05 else 1 if p_value < 0.10 else 0
        bins[index] = level if float(z_score) >= 0 else -level
    return bins


def getis_ord_class_summary(confidence_bins, missing=None) -> dict:
    """Summarize Getis-Ord Gi* confidence-bin output.

    ``missing`` maps "No data" / "No neighbours" to feature counts.  Those
    features carry ``gi_conf = 0`` (contract C2) and must be included in
    ``confidence_bins``; they are counted apart from the tested-and-not-
    significant ones so the report matches the map.
    """
    missing = {k: int(v) for k, v in (missing or {}).items() if v}
    counts = Counter(int(value) for value in confidence_bins if value is not None)
    if missing:
        counts[0] -= sum(missing.values())
        if counts[0] <= 0:
            del counts[0]
    hot_count = sum(count for klass, count in counts.items() if klass > 0)
    cold_count = sum(count for klass, count in counts.items() if klass < 0)
    significant_count = hot_count + cold_count
    dominant = _dominant_label(counts, GI_CONF_LABELS)
    return {
        "counts": dict(sorted(counts.items())),
        "missing": missing,
        "hot_count": int(hot_count),
        "cold_count": int(cold_count),
        "significant_count": int(significant_count),
        "tested_count": int(sum(counts.values())),
        "dominant_label": dominant,
        "message": _gi_message(hot_count, cold_count, significant_count, dominant) + missing_message(missing),
    }


def missing_message(missing: dict) -> str:
    """" Not tested: ..." sentence for the No data / No neighbours counts."""
    parts = []
    if missing.get(NO_DATA):
        parts.append(f"{int(missing[NO_DATA])} feature(s) had no usable value (No data)")
    if missing.get(NO_NEIGHBOURS):
        parts.append(f"{int(missing[NO_NEIGHBOURS])} feature(s) had no neighbours (No neighbours)")
    if not parts:
        return ""
    return " Not tested: " + "; ".join(parts) + "."


def local_moran_class_summary(quadrants) -> dict:
    """Summarize Local Moran cluster/outlier quadrant output."""
    normalized = [str(value) if value else "Not Significant" for value in quadrants]
    counts = Counter(normalized)
    missing = {klass: int(counts.pop(klass)) for klass in MISSING_CLASSES if counts.get(klass)}
    cluster_count = counts.get("HH", 0) + counts.get("LL", 0)
    outlier_count = counts.get("HL", 0) + counts.get("LH", 0)
    dominant = _dominant_label(counts, LISA_LABELS)
    all_counts = dict(sorted(counts.items()))
    all_counts.update(missing)
    return {
        "counts": all_counts,
        "missing": missing,
        "cluster_count": int(cluster_count),
        "outlier_count": int(outlier_count),
        "significant_count": int(cluster_count + outlier_count),
        "tested_count": int(sum(counts.values())),
        "dominant_label": dominant,
        "message": _lisa_message(cluster_count, outlier_count, dominant) + missing_message(missing),
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
