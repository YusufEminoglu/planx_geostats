# -*- coding: utf-8 -*-
"""Gradient Boosting Regression (CatBoost engine) Processing Algorithm."""
from __future__ import annotations

from ._gbm_base import GBMRegressionBase
from ._icons import algorithm_icon


class GBMRegressionCatBoostAlgorithm(GBMRegressionBase):
    ENGINE = "catboost"

    def name(self) -> str:
        return "gbm_regression_catboost"

    def displayName(self) -> str:
        return "Gradient Boosting Regression (CatBoost)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("gbm_regression_catboost")

    def createInstance(self):
        return GBMRegressionCatBoostAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Gradient Boosting regressor using CatBoost (Prokhorenkova et "
            "al., 2018) - ordered boosting with symmetric (oblivious) trees, "
            "designed from the ground up around native categorical-field "
            "handling. Requires the optional catboost package (Setup and "
            "Diagnostics > Install / Update GeoStats Libraries).\n\n"
            "Output: fitted values and residuals per complete record, plus an "
            "HTML report with R2, RMSE, MAE, and a native feature-importance "
            "table.\n\n"
            "CatBoost's headline difference from the other three engines is "
            "ordered boosting: instead of computing each tree's targets using "
            "residuals from a model fit on the same data (which can leak "
            "information and bias split selection, particularly with "
            "categorical fields encoded as target statistics), it computes each "
            "prediction using only a random permutation of records that "
            "precede it, avoiding the target leakage naive categorical "
            "encoding is prone to. This tool still requires numeric "
            "explanatory fields like every other regression tool in this "
            "group, but the engine's internal robustness to encoding-related "
            "leakage is still relevant background for interpreting how "
            "stable its fit is compared to the other engines.\n\n"
            "In-sample R2 is optimistic for the same reason as every boosting "
            "engine here - confirm with Spatial k-Fold Cross-Validation "
            "Evaluator. Compare against the other three engines on the same "
            "fields; CatBoost's symmetric-tree structure tends to generalize "
            "well with default settings, making it a reasonable first engine "
            "to try before tuning."
        )
