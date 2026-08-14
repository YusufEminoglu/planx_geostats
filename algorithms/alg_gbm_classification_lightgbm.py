# -*- coding: utf-8 -*-
"""Gradient Boosting Classification (LightGBM engine) Processing Algorithm."""
from __future__ import annotations

from ._gbm_base import GBMClassificationBase
from ._icons import algorithm_icon


class GBMClassificationLightGBMAlgorithm(GBMClassificationBase):
    ENGINE = "lightgbm"

    def name(self) -> str:
        return "gbm_classification_lightgbm"

    def displayName(self) -> str:
        return "Gradient Boosting Classification (LightGBM)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("gbm_classification_lightgbm")

    def createInstance(self):
        return GBMClassificationLightGBMAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Gradient Boosting classifier using LightGBM, grown leaf-wise "
            "(expanding whichever leaf reduces error most) instead of level-wise "
            "- usually faster and able to fit sharper local decision boundaries "
            "on larger datasets. Requires the optional lightgbm package (Setup "
            "and Diagnostics > Install / Update GeoStats Libraries).\n\n"
            "Output: predicted class and prediction confidence per complete "
            "record, plus an HTML report with accuracy, confusion matrix, "
            "per-class precision/recall/F1, and a native feature-importance "
            "table.\n\n"
            "Leaf-wise growth is the most overfitting-prone of the three engines "
            "on small datasets - reduce Max tree depth or lower the learning rate "
            "if in-sample accuracy looks suspiciously close to 1.0, and always "
            "confirm with Spatial k-Fold Cross-Validation Evaluator before "
            "reporting a final accuracy figure.\n\n"
            "Compare against Gradient Boosting Classification (scikit-learn) and "
            "(XGBoost) on the same fields; LightGBM tends to pull ahead on larger "
            "datasets and roughly match the others on small planning datasets."
        )
