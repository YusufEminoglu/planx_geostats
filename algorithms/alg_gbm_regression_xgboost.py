# -*- coding: utf-8 -*-
"""Gradient Boosting Regression (XGBoost engine) Processing Algorithm."""
from __future__ import annotations

from ._gbm_base import GBMRegressionBase
from ._icons import algorithm_icon


class GBMRegressionXGBoostAlgorithm(GBMRegressionBase):
    ENGINE = "xgboost"

    def name(self) -> str:
        return "gbm_regression_xgboost"

    def displayName(self) -> str:
        return "Gradient Boosting Regression (XGBoost)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("gbm_regression_xgboost")

    def createInstance(self):
        return GBMRegressionXGBoostAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Gradient Boosting regressor using XGBoost - trees added "
            "sequentially, each fit to the residual error of the ensemble so "
            "far, with second-order (Newton) boosting and built-in L1/L2 "
            "regularization. Requires the optional xgboost package (Setup and "
            "Diagnostics > Install / Update GeoStats Libraries).\n\n"
            "Output: fitted values and residuals per complete record, plus an "
            "HTML report with R2, RMSE, MAE, and a native ranked feature-"
            "importance table (gain-based).\n\n"
            "Like all boosting engines, XGBoost fits training data more closely "
            "than Random Forest at a given tree count - the in-sample R2 shown "
            "here is optimistic. Always confirm with Spatial k-Fold Cross-"
            "Validation Evaluator before reporting a final R2. XGBoost's built-in "
            "regularization makes it somewhat more overfitting-resistant than the "
            "scikit-learn engine at default settings, but a high learning rate "
            "with many boosting rounds is still the most common failure mode.\n\n"
            "Compare against Gradient Boosting Regression (scikit-learn) and "
            "(LightGBM) on the same fields - close agreement across engines is "
            "reassuring; large disagreement suggests the fit is sensitive to "
            "engine-specific defaults and should be tuned rather than trusted "
            "as-is."
        )
