# -*- coding: utf-8 -*-
"""Shared feature-matrix extraction and scikit-learn model wrappers for the
PlanX GeoStats Lab Machine Learning and Explainable AI group.

scikit-learn is imported lazily inside each fit_* function so this module
stays importable (and the Processing provider stays loadable) when the
optional package is missing; callers should wrap fit_* calls in the same
optional_dependency_error() pattern already used by GWR/MGWR/spreg tools.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from qgis.core import NULL


def _to_float(value) -> Optional[float]:
    if value is None or value == NULL or str(value) == "NULL":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def extract_regression_matrix(
    source,
    feature_fields: Sequence[str],
    target_field: str,
    feedback=None,
    progress_span: Tuple[int, int] = (0, 30),
) -> dict:
    """Extract a complete-case (X, y) matrix for a numeric target.

    Returns a dict with keys: x (ndarray[n, p]), y (ndarray[n]),
    valid_fids (list[int]), skipped (int), total (int).
    """
    fields = source.fields()
    target_idx = fields.lookupField(target_field)
    if target_idx < 0:
        raise ValueError(f"Target field '{target_field}' not found.")
    feature_indices = [fields.lookupField(name) for name in feature_fields]
    missing = [name for name, idx in zip(feature_fields, feature_indices) if idx < 0]
    if missing:
        raise ValueError(f"Feature fields not found: {', '.join(missing)}")

    start, end = progress_span
    total = source.featureCount() or 1
    x_rows: List[List[float]] = []
    y_values: List[float] = []
    valid_fids: List[int] = []
    skipped = 0

    for idx, feature in enumerate(source.getFeatures()):
        if feedback and feedback.isCanceled():
            break
        target_value = _to_float(feature.attribute(target_idx))
        row = [_to_float(feature.attribute(field_idx)) for field_idx in feature_indices]
        if target_value is None or any(value is None for value in row):
            skipped += 1
            continue
        y_values.append(target_value)
        x_rows.append(row)
        valid_fids.append(feature.id())
        if feedback:
            feedback.setProgress(int(start + (end - start) * (idx / total)))

    return {
        "x": np.array(x_rows, dtype=float),
        "y": np.array(y_values, dtype=float),
        "valid_fids": valid_fids,
        "skipped": skipped,
        "total": int(source.featureCount()),
    }


def extract_classification_matrix(
    source,
    feature_fields: Sequence[str],
    target_field: str,
    feedback=None,
    progress_span: Tuple[int, int] = (0, 30),
) -> dict:
    """Extract a complete-case (X, y) matrix for a categorical target.

    The target may be numeric or text; classes are the sorted set of
    distinct values seen among complete records. Returns a dict with keys:
    x, y (int class-index array), class_labels (list[str], sorted),
    valid_fids, skipped, total.
    """
    fields = source.fields()
    target_idx = fields.lookupField(target_field)
    if target_idx < 0:
        raise ValueError(f"Target field '{target_field}' not found.")
    feature_indices = [fields.lookupField(name) for name in feature_fields]
    missing = [name for name, idx in zip(feature_fields, feature_indices) if idx < 0]
    if missing:
        raise ValueError(f"Feature fields not found: {', '.join(missing)}")

    start, end = progress_span
    total = source.featureCount() or 1
    x_rows: List[List[float]] = []
    raw_targets: List[str] = []
    valid_fids: List[int] = []
    skipped = 0

    for idx, feature in enumerate(source.getFeatures()):
        if feedback and feedback.isCanceled():
            break
        target_value = feature.attribute(target_idx)
        row = [_to_float(feature.attribute(field_idx)) for field_idx in feature_indices]
        if target_value is None or target_value == NULL or str(target_value) == "NULL" or any(
            value is None for value in row
        ):
            skipped += 1
            continue
        raw_targets.append(str(target_value))
        x_rows.append(row)
        valid_fids.append(feature.id())
        if feedback:
            feedback.setProgress(int(start + (end - start) * (idx / total)))

    class_labels = sorted(set(raw_targets))
    label_to_index = {label: i for i, label in enumerate(class_labels)}
    y_indices = np.array([label_to_index[label] for label in raw_targets], dtype=int)

    return {
        "x": np.array(x_rows, dtype=float),
        "y": y_indices,
        "class_labels": class_labels,
        "valid_fids": valid_fids,
        "skipped": skipped,
        "total": int(source.featureCount()),
    }


def extract_matrix_with_centroids(
    source,
    feature_fields: Sequence[str],
    target_field: str,
    task_type: str,
    feedback=None,
    progress_span: Tuple[int, int] = (0, 30),
) -> dict:
    """Like extract_regression_matrix/extract_classification_matrix, but also
    returns each complete record's geometry centroid (x, y) for spatial
    cross-validation folding. task_type is 'regression' or 'classification'."""
    from .weights import geometry_centroid_point

    fields = source.fields()
    target_idx = fields.lookupField(target_field)
    if target_idx < 0:
        raise ValueError(f"Target field '{target_field}' not found.")
    feature_indices = [fields.lookupField(name) for name in feature_fields]
    missing = [name for name, idx in zip(feature_fields, feature_indices) if idx < 0]
    if missing:
        raise ValueError(f"Feature fields not found: {', '.join(missing)}")

    start, end = progress_span
    total = source.featureCount() or 1
    x_rows: List[List[float]] = []
    raw_targets: List = []
    centroid_rows: List[List[float]] = []
    valid_fids: List[int] = []
    skipped = 0

    for idx, feature in enumerate(source.getFeatures()):
        if feedback and feedback.isCanceled():
            break
        centroid = geometry_centroid_point(feature.geometry())
        raw_target = feature.attribute(target_idx)
        if task_type == "classification":
            target_missing = raw_target is None or raw_target == NULL or str(raw_target) == "NULL"
            target_value = str(raw_target) if not target_missing else None
        else:
            target_value = _to_float(raw_target)
            target_missing = target_value is None
        row = [_to_float(feature.attribute(field_idx)) for field_idx in feature_indices]
        if centroid is None or target_missing or any(value is None for value in row):
            skipped += 1
            continue
        x_rows.append(row)
        raw_targets.append(target_value)
        centroid_rows.append([centroid.x(), centroid.y()])
        valid_fids.append(feature.id())
        if feedback:
            feedback.setProgress(int(start + (end - start) * (idx / total)))

    x = np.array(x_rows, dtype=float)
    centroids = np.array(centroid_rows, dtype=float)
    result = {"x": x, "centroids": centroids, "valid_fids": valid_fids, "skipped": skipped, "total": int(source.featureCount())}
    if task_type == "classification":
        class_labels = sorted(set(raw_targets))
        label_to_index = {label: i for i, label in enumerate(class_labels)}
        result["y"] = np.array([label_to_index[label] for label in raw_targets], dtype=int)
        result["class_labels"] = class_labels
    else:
        result["y"] = np.array(raw_targets, dtype=float)
    return result


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    return {"r2": r2, "rmse": rmse, "mae": mae, "residuals": residuals}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_labels: Sequence[str]) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    n_classes = len(class_labels)
    accuracy = float(np.mean(y_true == y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    per_class = [
        {
            "label": class_labels[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(n_classes)
    ]
    return {"accuracy": accuracy, "confusion_matrix": cm, "per_class": per_class}


def _tree_ensemble_regressor(model_cls, x: np.ndarray, y: np.ndarray, **kwargs) -> dict:
    model = model_cls(**kwargs)
    model.fit(x, y)
    fitted = model.predict(x)
    metrics = regression_metrics(y, fitted)
    oob_score = float(model.oob_score_) if getattr(model, "oob_score", False) and hasattr(model, "oob_score_") else None
    return {
        "model": model,
        "fitted": fitted,
        "feature_importances": model.feature_importances_.tolist(),
        "oob_score": oob_score,
        **metrics,
    }


def _tree_ensemble_classifier(model_cls, x: np.ndarray, y: np.ndarray, class_labels: Sequence[str], **kwargs) -> dict:
    model = model_cls(**kwargs)
    model.fit(x, y)
    fitted = model.predict(x)
    proba = model.predict_proba(x)
    metrics = classification_metrics(y, fitted, class_labels)
    oob_score = float(model.oob_score_) if getattr(model, "oob_score", False) and hasattr(model, "oob_score_") else None
    return {
        "model": model,
        "fitted": fitted,
        "proba": proba,
        "feature_importances": model.feature_importances_.tolist(),
        "oob_score": oob_score,
        **metrics,
    }


def permutation_feature_importance(
    x: np.ndarray, y: np.ndarray, feature_names: Sequence[str], task_type: str, model_key: str,
    n_repeats: int = 10, random_state: int = 42,
) -> dict:
    """Fit the chosen model on the full data, then rank fields by how much a
    metric drops when that field's values are randomly shuffled - a
    model-agnostic importance measure that works even for SVM/MLP, which
    expose no native feature_importances_."""
    from sklearn.inspection import permutation_importance as sk_permutation_importance

    estimator = build_cv_estimator(task_type, model_key, random_state)
    estimator.fit(x, y)
    scoring = "r2" if task_type == "regression" else "accuracy"
    result = sk_permutation_importance(estimator, x, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring)
    rows = sorted(
        zip(feature_names, result.importances_mean.tolist(), result.importances_std.tolist()),
        key=lambda item: item[1], reverse=True,
    )
    return {"rows": rows, "baseline_score": float(estimator.score(x, y)), "scoring": scoring}


def partial_dependence_report(
    x: np.ndarray, y: np.ndarray, feature_index: int, task_type: str, model_key: str,
    grid_resolution: int = 20, random_state: int = 42,
) -> dict:
    """Fit the chosen model, then compute the 1D partial dependence curve for
    one feature: predictions averaged over all records while that feature is
    swept across a grid of values, holding every other field at its observed
    values (Friedman's PDP). For classification this reports the effect on
    the predicted probability of the first class in sorted class-label order."""
    from sklearn.inspection import partial_dependence

    estimator = build_cv_estimator(task_type, model_key, random_state)
    estimator.fit(x, y)
    pd_result = partial_dependence(estimator, x, [feature_index], grid_resolution=grid_resolution, kind="average")
    grid_values = pd_result.get("grid_values", pd_result.get("values"))[0]
    averaged = np.asarray(pd_result["average"])[0]
    return {"grid_values": grid_values.tolist(), "average": averaged.tolist()}


def fit_random_forest_regressor(
    x: np.ndarray, y: np.ndarray, n_estimators: int = 200, max_depth: int = 0,
    min_samples_leaf: int = 1, random_state: int = 42,
) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    return _tree_ensemble_regressor(
        RandomForestRegressor, x, y,
        n_estimators=n_estimators, max_depth=(max_depth or None),
        min_samples_leaf=min_samples_leaf, random_state=random_state,
        oob_score=True, bootstrap=True, n_jobs=-1,
    )


def fit_random_forest_classifier(
    x: np.ndarray, y: np.ndarray, class_labels: Sequence[str], n_estimators: int = 200,
    max_depth: int = 0, min_samples_leaf: int = 1, random_state: int = 42,
) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    return _tree_ensemble_classifier(
        RandomForestClassifier, x, y, class_labels,
        n_estimators=n_estimators, max_depth=(max_depth or None),
        min_samples_leaf=min_samples_leaf, random_state=random_state,
        oob_score=True, bootstrap=True, n_jobs=-1,
    )


def fit_extra_trees_regressor(
    x: np.ndarray, y: np.ndarray, n_estimators: int = 200, max_depth: int = 0,
    min_samples_leaf: int = 1, random_state: int = 42,
) -> dict:
    from sklearn.ensemble import ExtraTreesRegressor
    return _tree_ensemble_regressor(
        ExtraTreesRegressor, x, y,
        n_estimators=n_estimators, max_depth=(max_depth or None),
        min_samples_leaf=min_samples_leaf, random_state=random_state,
        oob_score=True, bootstrap=True, n_jobs=-1,
    )


def fit_extra_trees_classifier(
    x: np.ndarray, y: np.ndarray, class_labels: Sequence[str], n_estimators: int = 200,
    max_depth: int = 0, min_samples_leaf: int = 1, random_state: int = 42,
) -> dict:
    from sklearn.ensemble import ExtraTreesClassifier
    return _tree_ensemble_classifier(
        ExtraTreesClassifier, x, y, class_labels,
        n_estimators=n_estimators, max_depth=(max_depth or None),
        min_samples_leaf=min_samples_leaf, random_state=random_state,
        oob_score=True, bootstrap=True, n_jobs=-1,
    )


UNCERTAINTY_MODEL_KEYS = ["random_forest", "extra_trees"]


def prediction_uncertainty(
    x: np.ndarray, y: np.ndarray, model_key: str, n_estimators: int = 200, max_depth: int = 0,
    min_samples_leaf: int = 1, random_state: int = 42,
) -> dict:
    """Bagging-ensemble prediction uncertainty: the spread of individual tree
    predictions around their mean, for Random Forest / Extra Trees only -
    boosting and kernel/neural models do not have an equivalent per-member
    prediction spread without extra machinery, so this is intentionally
    scoped to the two bagging ensembles in this group."""
    if model_key == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=(max_depth or None), min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=-1,
        )
    elif model_key == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(
            n_estimators=n_estimators, max_depth=(max_depth or None), min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=-1,
        )
    else:
        raise ValueError(f"Prediction uncertainty is only available for: {UNCERTAINTY_MODEL_KEYS}")

    model.fit(x, y)
    per_tree = np.array([tree.predict(x) for tree in model.estimators_])
    return {
        "mean": per_tree.mean(axis=0),
        "std": per_tree.std(axis=0),
        "lower_10": np.percentile(per_tree, 10, axis=0),
        "upper_90": np.percentile(per_tree, 90, axis=0),
    }


def fit_svr(x: np.ndarray, y: np.ndarray, kernel: str = "rbf", c: float = 1.0, epsilon: float = 0.1) -> dict:
    """Support Vector Regression. Features and target are standardized internally
    (SVM is scale-sensitive); the returned model is a fitted sklearn Pipeline."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    y_mean, y_std = float(np.mean(y)), float(np.std(y)) or 1.0
    model = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=c, epsilon=epsilon))
    model.fit(x, (y - y_mean) / y_std)
    fitted = model.predict(x) * y_std + y_mean
    metrics = regression_metrics(y, fitted)
    return {"model": model, "fitted": fitted, "y_mean": y_mean, "y_std": y_std, **metrics}


def fit_svc(x: np.ndarray, y: np.ndarray, class_labels: Sequence[str], kernel: str = "rbf", c: float = 1.0) -> dict:
    """Support Vector Classification. Features are standardized internally."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c, probability=True, random_state=42))
    model.fit(x, y)
    fitted = model.predict(x)
    proba = model.predict_proba(x)
    metrics = classification_metrics(y, fitted, class_labels)
    return {"model": model, "fitted": fitted, "proba": proba, **metrics}


def fit_dbscan(x: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> dict:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    x_scaled = StandardScaler().fit_transform(x)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x_scaled)
    n_clusters = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
    return {"labels": labels, "n_clusters": n_clusters, "n_noise": int(np.sum(labels == -1))}


def fit_hdbscan(x: np.ndarray, min_cluster_size: int = 5, min_samples: Optional[int] = None) -> dict:
    """Requires scikit-learn >= 1.3 (sklearn.cluster.HDBSCAN)."""
    from sklearn.cluster import HDBSCAN
    from sklearn.preprocessing import StandardScaler

    x_scaled = StandardScaler().fit_transform(x)
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = model.fit_predict(x_scaled)
    probabilities = getattr(model, "probabilities_", np.ones(len(labels)))
    n_clusters = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
    return {"labels": labels, "probabilities": probabilities, "n_clusters": n_clusters, "n_noise": int(np.sum(labels == -1))}


def fit_gmm(x: np.ndarray, n_components: int = 3, covariance_type: str = "full", random_state: int = 42) -> dict:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    x_scaled = StandardScaler().fit_transform(x)
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    model.fit(x_scaled)
    labels = model.predict(x_scaled)
    proba = model.predict_proba(x_scaled)
    return {
        "labels": labels, "proba": proba, "bic": float(model.bic(x_scaled)), "aic": float(model.aic(x_scaled)),
        "converged": bool(model.converged_),
    }


def top_feature_importance_rows(feature_names: Sequence[str], importances: Iterable[float], limit: int = 20) -> List[Tuple[str, float]]:
    pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    return pairs[:limit]


GBM_ENGINES = ["sklearn", "xgboost", "lightgbm"]
GBM_ENGINE_LABELS = {
    "sklearn": "scikit-learn (HistGradientBoosting - no extra install)",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
GBM_ENGINE_PACKAGES = {"sklearn": ["scikit-learn"], "xgboost": ["xgboost"], "lightgbm": ["lightgbm"]}


def fit_gbm_regressor(
    engine: str, x: np.ndarray, y: np.ndarray, n_estimators: int = 200, learning_rate: float = 0.1,
    max_depth: int = 3, random_state: int = 42,
) -> dict:
    if engine == "sklearn":
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=n_estimators, learning_rate=learning_rate, max_depth=(max_depth or None), random_state=random_state,
        )
        model.fit(x, y)
        fitted = model.predict(x)
        return {"model": model, "fitted": fitted, "feature_importances": None, **regression_metrics(y, fitted)}
    if engine == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=random_state, verbosity=0,
        )
        model.fit(x, y)
        fitted = model.predict(x)
        return {
            "model": model, "fitted": fitted, "feature_importances": model.feature_importances_.tolist(),
            **regression_metrics(y, fitted),
        }
    if engine == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=(max_depth or -1),
            random_state=random_state, verbosity=-1,
        )
        model.fit(x, y)
        fitted = model.predict(x)
        return {
            "model": model, "fitted": fitted, "feature_importances": model.feature_importances_.tolist(),
            **regression_metrics(y, fitted),
        }
    raise ValueError(f"Unknown Gradient Boosting engine: {engine}")


def fit_gbm_classifier(
    engine: str, x: np.ndarray, y: np.ndarray, class_labels: Sequence[str], n_estimators: int = 200,
    learning_rate: float = 0.1, max_depth: int = 3, random_state: int = 42,
) -> dict:
    if engine == "sklearn":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=n_estimators, learning_rate=learning_rate, max_depth=(max_depth or None), random_state=random_state,
        )
        model.fit(x, y)
        fitted = model.predict(x)
        proba = model.predict_proba(x)
        return {
            "model": model, "fitted": fitted, "proba": proba, "feature_importances": None,
            **classification_metrics(y, fitted, class_labels),
        }
    if engine == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=random_state, verbosity=0, eval_metric="mlogloss" if len(class_labels) > 2 else "logloss",
        )
        model.fit(x, y)
        fitted = model.predict(x)
        proba = model.predict_proba(x)
        return {
            "model": model, "fitted": fitted, "proba": proba, "feature_importances": model.feature_importances_.tolist(),
            **classification_metrics(y, fitted, class_labels),
        }
    if engine == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=(max_depth or -1),
            random_state=random_state, verbosity=-1,
        )
        model.fit(x, y)
        fitted = model.predict(x)
        proba = model.predict_proba(x)
        return {
            "model": model, "fitted": fitted, "proba": proba, "feature_importances": model.feature_importances_.tolist(),
            **classification_metrics(y, fitted, class_labels),
        }
    raise ValueError(f"Unknown Gradient Boosting engine: {engine}")


def fit_mlp_regressor(
    x: np.ndarray, y: np.ndarray, hidden_layer_sizes: Tuple[int, ...] = (64, 32), max_iter: int = 1000,
    random_state: int = 42,
) -> dict:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_mean, y_std = float(np.mean(y)), float(np.std(y)) or 1.0
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    model = make_pipeline(StandardScaler(), mlp)
    model.fit(x, (y - y_mean) / y_std)
    fitted = model.predict(x) * y_std + y_mean
    converged = bool(mlp.n_iter_ < max_iter)
    return {
        "model": model, "fitted": fitted, "y_mean": y_mean, "y_std": y_std, "converged": converged,
        "n_iter": int(mlp.n_iter_), **regression_metrics(y, fitted),
    }


CV_MODEL_KEYS = ["random_forest", "extra_trees", "gbm", "svm", "mlp"]
CV_MODEL_LABELS = {
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gbm": "Gradient Boosting (scikit-learn)",
    "svm": "Support Vector Machine",
    "mlp": "Neural Network (MLP)",
}


def build_cv_estimator(task_type: str, model_key: str, random_state: int = 42):
    """Return an unfitted sklearn-compatible estimator for spatial cross-validation."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if task_type not in {"regression", "classification"}:
        raise ValueError(f"Unknown task type: {task_type}")
    is_regression = task_type == "regression"

    if model_key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestRegressor if is_regression else RandomForestClassifier
        return cls(n_estimators=200, random_state=random_state, n_jobs=-1)
    if model_key == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        cls = ExtraTreesRegressor if is_regression else ExtraTreesClassifier
        return cls(n_estimators=200, random_state=random_state, n_jobs=-1)
    if model_key == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        cls = HistGradientBoostingRegressor if is_regression else HistGradientBoostingClassifier
        return cls(random_state=random_state)
    if model_key == "svm":
        from sklearn.svm import SVC, SVR
        estimator = SVR() if is_regression else SVC(probability=False, random_state=random_state)
        return make_pipeline(StandardScaler(), estimator)
    if model_key == "mlp":
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        cls = MLPRegressor if is_regression else MLPClassifier
        return make_pipeline(StandardScaler(), cls(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=random_state))
    raise ValueError(f"Unknown model: {model_key}")


def spatial_kfold_assignment(centroids: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """Assign each record to one of k spatially contiguous folds via KMeans on centroid coordinates.

    Ordinary random k-fold CV leaks information under spatial autocorrelation
    (nearby train/test points are not independent); grouping geographically
    close records into the same fold gives an honest out-of-sample estimate.
    """
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return model.fit_predict(centroids)


def spatial_kfold_evaluate(
    x: np.ndarray, y: np.ndarray, centroids: np.ndarray, k: int, task_type: str, model_key: str,
    random_state: int = 42,
) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    folds = spatial_kfold_assignment(centroids, k, random_state)
    fold_metrics = []
    for fold_id in range(k):
        test_mask = folds == fold_id
        train_mask = ~test_mask
        if int(test_mask.sum()) == 0 or int(train_mask.sum()) == 0:
            continue
        estimator = build_cv_estimator(task_type, model_key, random_state)
        estimator.fit(x[train_mask], y[train_mask])
        pred = estimator.predict(x[test_mask])
        if task_type == "regression":
            metrics = regression_metrics(y[test_mask], pred)
            fold_metrics.append({
                "fold": fold_id, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
                "r2": metrics["r2"], "rmse": metrics["rmse"], "mae": metrics["mae"],
            })
        else:
            fold_metrics.append({
                "fold": fold_id, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
                "accuracy": float(accuracy_score(y[test_mask], pred)),
                "macro_f1": float(f1_score(y[test_mask], pred, average="macro", zero_division=0)),
            })
    return {"folds": folds, "fold_metrics": fold_metrics}


def fit_mlp_classifier(
    x: np.ndarray, y: np.ndarray, class_labels: Sequence[str], hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    max_iter: int = 1000, random_state: int = 42,
) -> dict:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    model = make_pipeline(StandardScaler(), mlp)
    model.fit(x, y)
    fitted = model.predict(x)
    proba = model.predict_proba(x)
    converged = bool(mlp.n_iter_ < max_iter)
    return {
        "model": model, "fitted": fitted, "proba": proba, "converged": converged, "n_iter": int(mlp.n_iter_),
        **classification_metrics(y, fitted, class_labels),
    }
