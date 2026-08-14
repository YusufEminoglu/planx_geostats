# -*- coding: utf-8 -*-
"""Gradient Boosting Regression (scikit-learn engine) Processing Algorithm."""
from __future__ import annotations

from ._gbm_base import GBMRegressionBase
from ._icons import algorithm_icon


class GBMRegressionSklearnAlgorithm(GBMRegressionBase):
    ENGINE = "sklearn"

    def name(self) -> str:
        return "gbm_regression_sklearn"

    def displayName(self) -> str:
        return "Gradient Boosting Regression (scikit-learn)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("gbm_regression_sklearn")

    def createInstance(self):
        return GBMRegressionSklearnAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Gradient Boosting regressor using scikit-learn's "
            "HistGradientBoostingRegressor - trees added sequentially, each fit "
            "to the residual error of the ensemble so far, using histogram-binned "
            "features for speed. Runs with no extra package beyond scikit-learn "
            "(already required by several other GeoStats tools), so this is the "
            "zero-extra-install way to try Gradient Boosting.\n\n"
            "Output: fitted values and residuals per complete record, plus an "
            "HTML report with R2, RMSE, and MAE. This engine does not expose "
            "per-field feature importances directly - run Permutation Feature "
            "Importance afterward on the same field selection if you need a "
            "ranked importance table; for that, XGBoost or LightGBM report "
            "importances natively.\n\n"
            "Gradient Boosting fits training data more closely than Random Forest "
            "at a given tree count, which means the in-sample R2 shown here is "
            "more optimistic than usual - always confirm with Spatial k-Fold "
            "Cross-Validation Evaluator before reporting an R2 figure. Lower the "
            "learning rate and raise the number of boosting rounds together for a "
            "more conservative fit; a high learning rate with many rounds is the "
            "most common cause of severe overfitting here."
        )
