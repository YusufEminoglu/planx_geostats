# -*- coding: utf-8 -*-
"""Gradient Boosting Classification (scikit-learn engine) Processing Algorithm."""
from __future__ import annotations

from ._gbm_base import GBMClassificationBase
from ._icons import algorithm_icon


class GBMClassificationSklearnAlgorithm(GBMClassificationBase):
    ENGINE = "sklearn"

    def name(self) -> str:
        return "gbm_classification_sklearn"

    def displayName(self) -> str:
        return "Gradient Boosting Classification (scikit-learn)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("gbm_classification_sklearn")

    def createInstance(self):
        return GBMClassificationSklearnAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Gradient Boosting classifier using scikit-learn's "
            "HistGradientBoostingClassifier - trees added sequentially, each "
            "correcting the ensemble's classification errors, using histogram-"
            "binned features for speed. Runs with no extra package beyond "
            "scikit-learn, so this is the zero-extra-install way to try Gradient "
            "Boosting classification.\n\n"
            "Output: predicted class and prediction confidence per complete "
            "record, plus an HTML report with accuracy, confusion matrix, and "
            "per-class precision/recall/F1. This engine does not expose "
            "per-field feature importances directly - run Permutation Feature "
            "Importance afterward, or use XGBoost/LightGBM for native importances.\n\n"
            "Boosting classifiers fit training data closely, so in-sample "
            "accuracy here is optimistic - confirm with Spatial k-Fold Cross-"
            "Validation Evaluator before reporting a final accuracy figure, "
            "especially with imbalanced classes where per-class recall matters "
            "more than overall accuracy."
        )
