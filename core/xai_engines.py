# -*- coding: utf-8 -*-
"""SHAP-based explainability wrappers for the PlanX GeoStats Lab Machine
Learning and Explainable AI group.

shap is imported lazily so this module (and the Processing provider) stays
importable when the optional package is missing; callers should wrap calls
in the same optional_dependency_error() pattern already used elsewhere.

SHAP's return shapes differ across library versions and explainer types
(TreeExplainer historically returned a list of per-class arrays for
classifiers; newer releases return a single array with a trailing class
axis). _normalize_shap_output() below handles both documented shapes; this
was implemented against SHAP's published API and should be re-checked
against the shap version actually resolved in the QGIS Python environment
before this tool group ships (see the Faz 5 OSGeo4W runtime gate).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .ml_engines import build_cv_estimator

TREE_MODEL_KEYS = {"random_forest", "extra_trees", "gbm"}


def _normalize_shap_output(raw, task_type: str, class_index: int) -> np.ndarray:
    """Return a plain (n_samples, n_features) array for the requested class
    (classification) or the single output (regression), regardless of which
    SHAP return-shape convention the installed version used."""
    if isinstance(raw, list):
        # Legacy convention: one (n_samples, n_features) array per class.
        if task_type == "classification":
            return np.asarray(raw[class_index])
        return np.asarray(raw[0])

    array = np.asarray(raw)
    if task_type == "classification" and array.ndim == 3:
        # (n_samples, n_features, n_classes)
        return array[:, :, class_index]
    if array.ndim == 3:
        # Regression explainer that still returned a trailing singleton axis.
        return array[:, :, 0]
    return array


def compute_shap_values(
    x: np.ndarray, y: np.ndarray, task_type: str, model_key: str, class_index: int = 0,
    max_rows: int = 200, background_size: int = 50, random_state: int = 42,
    explicit_rows: Optional[Sequence[int]] = None,
) -> dict:
    """Fit the chosen model and compute SHAP values for up to max_rows records
    (a random subsample when the input is larger, to keep KernelExplainer/
    PermutationExplainer runtime bounded for non-tree models). Pass
    explicit_rows to explain exactly those row indices instead (e.g. a single
    record picked by feature ID for a local explanation) - background
    sampling for the black-box explainer still draws from the full dataset.

    Returns: sample_idx (indices into the original x/y arrays that were
    explained), shap_values (n_sample, n_features), base_value (float).
    """
    import shap

    estimator = build_cv_estimator(task_type, model_key, random_state)
    estimator.fit(x, y)

    n = x.shape[0]
    rng = np.random.default_rng(random_state)
    if explicit_rows is not None:
        sample_idx = np.asarray(explicit_rows)
    else:
        sample_idx = np.sort(rng.choice(n, size=min(max_rows, n), replace=False)) if n > max_rows else np.arange(n)
    x_sample = x[sample_idx]
    background_n = min(background_size, n)
    background = x[np.sort(rng.choice(n, size=background_n, replace=False))]

    if model_key in TREE_MODEL_KEYS:
        explainer = shap.TreeExplainer(estimator)
        raw = explainer.shap_values(x_sample)
        base = explainer.expected_value
    else:
        predict_fn = estimator.predict_proba if task_type == "classification" else estimator.predict
        explainer = shap.Explainer(predict_fn, background)
        explanation = explainer(x_sample)
        raw = explanation.values
        base = explanation.base_values

    values = _normalize_shap_output(raw, task_type, class_index)

    if isinstance(base, (list, np.ndarray)):
        base_array = np.asarray(base)
        if base_array.ndim >= 1 and base_array.shape[-1] > 1 and task_type == "classification":
            base_value = float(np.ravel(base_array)[class_index])
        else:
            base_value = float(np.mean(base_array))
    else:
        base_value = float(base)

    return {"sample_idx": sample_idx, "shap_values": values, "base_value": base_value}


def shap_global_importance(shap_values: np.ndarray, feature_names: Sequence[str]) -> List[Tuple[str, float]]:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    return sorted(zip(feature_names, mean_abs.tolist()), key=lambda item: item[1], reverse=True)


def shap_local_breakdown(
    shap_values: np.ndarray, feature_names: Sequence[str], row_index: int, base_value: float,
) -> dict:
    row = shap_values[row_index]
    contributions = sorted(zip(feature_names, row.tolist()), key=lambda item: abs(item[1]), reverse=True)
    return {
        "base_value": base_value,
        "contributions": contributions,
        "prediction": float(base_value + np.sum(row)),
    }
