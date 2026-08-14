# -*- coding: utf-8 -*-
"""Gradient Boosting Classification (XGBoost engine) Processing Algorithm."""
from __future__ import annotations

from ._gbm_base import GBMClassificationBase
from ._icons import algorithm_icon


class GBMClassificationXGBoostAlgorithm(GBMClassificationBase):
    ENGINE = "xgboost"

    def name(self) -> str:
        return "gbm_classification_xgboost"

    def displayName(self) -> str:
        return "Gradient Boosting Classification (XGBoost)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("gbm_classification_xgboost")

    def createInstance(self):
        return GBMClassificationXGBoostAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Gradient Boosting classifier using XGBoost - trees added "
            "sequentially with second-order (Newton) boosting and built-in "
            "L1/L2 regularization. Requires the optional xgboost package (Setup "
            "and Diagnostics > Install / Update GeoStats Libraries).\n\n"
            "Output: predicted class and prediction confidence per complete "
            "record, plus an HTML report with accuracy, confusion matrix, "
            "per-class precision/recall/F1, and a native gain-based "
            "feature-importance table.\n\n"
            "In-sample accuracy is optimistic for the same reason as every "
            "boosting engine - confirm with Spatial k-Fold Cross-Validation "
            "Evaluator, and watch per-class recall rather than overall accuracy "
            "when classes are imbalanced. XGBoost's built-in regularization "
            "makes it somewhat more overfitting-resistant than the scikit-learn "
            "engine at default settings.\n\n"
            "Compare against Gradient Boosting Classification (scikit-learn) and "
            "(LightGBM) on the same fields; strong agreement across engines is "
            "reassuring evidence the classification pattern is real."
        )
