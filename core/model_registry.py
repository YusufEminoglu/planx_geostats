# -*- coding: utf-8 -*-
"""Safe, session-scoped registry for reusing fitted estimators.

Estimators stay in memory; only an auditable JSON manifest is written.  This
deliberately avoids pickle/joblib deserialization in a QGIS plugin.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


_MODELS = {}


def training_fingerprint(x: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    digest.update("\x1f".join(feature_names).encode("utf-8"))
    return digest.hexdigest()


def register_fitted_model(
    estimator, x: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
    target_field: str, task_type: str, model_key: str,
    class_labels: Optional[Sequence[str]] = None, metrics: Optional[dict] = None,
    manifest_path: Optional[str] = None,
) -> dict:
    fingerprint = training_fingerprint(x, y, feature_names)
    model_id = f"{model_key}-{fingerprint[:16]}"
    manifest = {
        "schema": "planx-geostats-model-audit-v1",
        "model_id": model_id,
        "model_key": model_key,
        "task_type": task_type,
        "target_field": target_field,
        "feature_fields": list(feature_names),
        "class_labels": list(class_labels or []),
        "record_count": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "training_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {key: value for key, value in (metrics or {}).items() if isinstance(value, (str, int, float, bool, type(None)))},
        "persistence": "estimator is reusable in this QGIS session; JSON is an audit manifest, not executable model data",
    }
    _MODELS[model_id] = {"estimator": estimator, "manifest": manifest}
    if manifest_path:
        Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def resolve_fitted_model(
    model_id: str, task_type: str, feature_names: Sequence[str],
    x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
):
    entry = _MODELS.get((model_id or "").strip())
    if entry is None:
        raise ValueError("Saved model ID is not available in this QGIS session. Run its training tool first.")
    manifest = entry["manifest"]
    if manifest["task_type"] != task_type:
        raise ValueError(f"Saved model task is {manifest['task_type']}, not {task_type}.")
    if manifest["feature_fields"] != list(feature_names):
        raise ValueError("Saved model explanatory fields or their order do not match this run.")
    if x is not None and y is not None:
        current = training_fingerprint(x, y, feature_names)
        if current != manifest["training_fingerprint"]:
            raise ValueError("Saved model training records differ from this run; refusing unaudited reuse.")
    return entry["estimator"], manifest


def clear_session_models() -> None:
    _MODELS.clear()
