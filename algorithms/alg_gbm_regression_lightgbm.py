# -*- coding: utf-8 -*-
"""Gradient Boosting Regression (LightGBM engine) Processing Algorithm."""
from __future__ import annotations

from ._gbm_base import GBMRegressionBase
from ._icons import algorithm_icon


class GBMRegressionLightGBMAlgorithm(GBMRegressionBase):
    ENGINE = "lightgbm"

    def name(self) -> str:
        return "gbm_regression_lightgbm"

    def displayName(self) -> str:
        return "Gradient Boosting Regression (LightGBM)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("gbm_regression_lightgbm")

    def createInstance(self):
        return GBMRegressionLightGBMAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Gradient Boosting regressor using LightGBM - trees added "
            "sequentially like the other two engines, but grown leaf-wise "
            "(expanding whichever leaf reduces error most) instead of level-wise, "
            "which is usually faster and can fit sharper local patterns on larger "
            "datasets. Requires the optional lightgbm package (Setup and "
            "Diagnostics > Install / Update GeoStats Libraries).\n\n"
            "Output: fitted values and residuals per complete record, plus an "
            "HTML report with R2, RMSE, MAE, and a native ranked feature-"
            "importance table.\n\n"
            "Leaf-wise growth fits training data very closely and is the most "
            "overfitting-prone of the three engines on small datasets (a few "
            "hundred records or fewer) - reduce Max tree depth or lower the "
            "learning rate if the in-sample R2 looks suspiciously close to 1.0. "
            "Always confirm with Spatial k-Fold Cross-Validation Evaluator before "
            "reporting an R2 figure.\n\n"
            "Compare against Gradient Boosting Regression (scikit-learn) and "
            "(XGBoost) on the same fields; LightGBM tends to pull ahead on larger "
            "datasets and roughly match the others on small planning datasets "
            "typical of a single district or city."
        )
