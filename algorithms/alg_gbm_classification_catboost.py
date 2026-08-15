# -*- coding: utf-8 -*-
"""Gradient Boosting Classification (CatBoost engine) Processing Algorithm."""
from __future__ import annotations

from ._gbm_base import GBMClassificationBase
from ._icons import algorithm_icon


class GBMClassificationCatBoostAlgorithm(GBMClassificationBase):
    ENGINE = "catboost"

    def name(self) -> str:
        return "gbm_classification_catboost"

    def displayName(self) -> str:
        return "Gradient Boosting Classification (CatBoost)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("gbm_classification_catboost")

    def createInstance(self):
        return GBMClassificationCatBoostAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Gradient Boosting classifier using CatBoost (Prokhorenkova "
            "et al., 2018) - ordered boosting with symmetric (oblivious) trees. "
            "Requires the optional catboost package (Setup and Diagnostics > "
            "Install / Update GeoStats Libraries).\n\n"
            "Output: predicted class and prediction confidence per complete "
            "record, plus an HTML report with accuracy, confusion matrix, "
            "per-class precision/recall/F1, and a native feature-importance "
            "table.\n\n"
            "CatBoost's ordered-boosting scheme specifically targets the "
            "target-leakage risk that naive categorical-field encoding "
            "introduces into gradient boosting - each prediction during "
            "training uses only a random permutation of preceding records, "
            "not the full dataset's target statistics. This tool still takes "
            "numeric explanatory fields like every other classifier in this "
            "group, but the engine's design still tends to produce stable, "
            "well-calibrated fits with default settings.\n\n"
            "In-sample accuracy is optimistic for the same reason as every "
            "boosting engine here - confirm with Spatial k-Fold Cross-"
            "Validation Evaluator, and watch per-class recall rather than "
            "overall accuracy when classes are imbalanced. Compare against "
            "the scikit-learn, XGBoost, and LightGBM engines on the same "
            "fields for a cross-engine sanity check."
        )
