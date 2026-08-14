# -*- coding: utf-8 -*-
"""SHAP Global Feature Importance Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

from ._mixins import HelpUrlMixin
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)

from ..core.ml_engines import CV_MODEL_KEYS, CV_MODEL_LABELS, extract_classification_matrix, extract_regression_matrix
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.xai_engines import compute_shap_values, shap_global_importance
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class SHAPGlobalImportanceAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    TASK_TYPE = "TASK_TYPE"
    MODEL = "MODEL"
    CLASS_LABEL = "CLASS_LABEL"
    MAX_ROWS = "MAX_ROWS"
    HTML_REPORT = "HTML_REPORT"

    TASK_TYPES = ["Regression (numeric target)", "Classification (categorical target)"]
    TASK_KEYS = ["regression", "classification"]
    MODEL_LABELS = [CV_MODEL_LABELS[key] for key in CV_MODEL_KEYS]

    def name(self) -> str:
        return "shap_global_importance"

    def displayName(self) -> str:
        return "SHAP Global Feature Importance"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("shap_global_importance")

    def createInstance(self):
        return SHAPGlobalImportanceAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits the chosen model and computes SHAP (SHapley Additive "
            "exPlanations) values: a game-theoretic attribution that splits "
            "each individual prediction into per-field contributions that sum "
            "exactly to (prediction - average prediction), fairly accounting "
            "for interactions between fields. Averaging the absolute "
            "contribution of each field across records gives a global "
            "importance ranking that, unlike Permutation Feature Importance, "
            "is also internally consistent with the local, per-record "
            "explanations produced by SHAP Local Explanation Report.\n\n"
            "Tree-based models (Random Forest, Extra Trees, Gradient Boosting) "
            "use the fast, exact TreeExplainer; Support Vector Machine and "
            "Neural Network use a background-sample-based explainer that "
            "treats the model as a black box and is slower - Max rows to "
            "explain caps runtime by explaining a random subsample rather "
            "than every record for large layers.\n\n"
            "For classification, SHAP values are computed for one class at a "
            "time - set Class to explain, or leave it blank to use the first "
            "class in sorted label order.\n\n"
            "Output: an HTML report ranking fields by mean absolute SHAP "
            "value. Follow up with SHAP Spatial Attribution Map to see where "
            "each field's contribution is concentrated geographically, or "
            "SHAP Local Explanation Report to see the full breakdown for one "
            "specific record."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.TARGET, "Target field", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FEATURES, "Explanatory fields (numeric)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric, allowMultiple=True,
            )
        )
        self.addParameter(QgsProcessingParameterEnum(self.TASK_TYPE, "Task type", options=self.TASK_TYPES, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.MODEL, "Model", options=self.MODEL_LABELS, defaultValue=0))
        self.addParameter(
            QgsProcessingParameterString(
                self.CLASS_LABEL, "Class to explain (classification only; blank = first class alphabetically)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ROWS, "Max rows to explain", type=QgsProcessingParameterNumber.Integer,
                defaultValue=200, minValue=20, maxValue=5000,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output SHAP global importance HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "SHAP global importance report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        task_type = self.TASK_KEYS[self.parameterAsEnum(parameters, self.TASK_TYPE, context)]
        model_key = CV_MODEL_KEYS[self.parameterAsEnum(parameters, self.MODEL, context)]
        class_label = self.parameterAsString(parameters, self.CLASS_LABEL, context)
        max_rows = self.parameterAsInt(parameters, self.MAX_ROWS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_shap_global_report.html")

        feedback.pushInfo("Extracting complete records...")
        try:
            if task_type == "regression":
                extraction = extract_regression_matrix(source, feature_fields, target_field, feedback, (0, 20))
            else:
                extraction = extract_classification_matrix(source, feature_fields, target_field, feedback, (0, 20))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, skipped = extraction["x"], extraction["y"], extraction["skipped"]
        if len(y) <= len(feature_fields) + 1:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {len(feature_fields)} explanatory field(s)."
            )
        class_index = 0
        class_labels = extraction.get("class_labels")
        if task_type == "classification":
            if len(class_labels) < 2:
                raise QgsProcessingException("At least 2 distinct classes are required for classification.")
            if class_label:
                if class_label not in class_labels:
                    raise QgsProcessingException(f"Class '{class_label}' not found among: {', '.join(class_labels)}")
                class_index = class_labels.index(class_label)
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing values.")

        feedback.pushInfo(f"Fitting {CV_MODEL_LABELS[model_key]} and computing SHAP values...")
        try:
            shap_result = compute_shap_values(x, y, task_type, model_key, class_index=class_index, max_rows=max_rows)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("SHAP Global Feature Importance", ["shap"], exc))

        importance_rows = shap_global_importance(shap_result["shap_values"], feature_fields)
        feedback.pushInfo(f"Explained {len(shap_result['sample_idx'])} of {len(y)} complete records.")

        explained_class = class_labels[class_index] if task_type == "classification" else None
        self._write_html(html_path, target_field, model_key, explained_class, len(shap_result["sample_idx"]), len(y), importance_rows, skipped)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, target_field, model_key, explained_class, n_explained, n_total, importance_rows, skipped):
        rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{value:.6f}</td></tr>" for name, value in importance_rows
        )
        class_note = f" | Class explained: <strong>{html.escape(explained_class)}</strong>" if explained_class else ""
        guidance = analyst_guidance_html(
            "SHAP Global Feature Importance",
            "Game-theoretic attribution: each field's contribution to every "
            "prediction sums exactly to that prediction's deviation from the "
            "average, fairly crediting field interactions.",
            [
                "n_explained is close to the full complete-record count (a small subsample increases noise in the ranking).",
                "The model fits reasonably well (SHAP explains what the model learned, not ground truth).",
            ],
            [
                "n_explained is a small fraction of complete records (raise Max rows to explain if runtime allows).",
                "Ranking disagrees sharply with Permutation Feature Importance on the same fields (worth investigating why).",
            ],
            [
                "SHAP Spatial Attribution Map - map where each field's contribution concentrates.",
                "SHAP Local Explanation Report - full breakdown for one specific record.",
            ],
            "Use this ranking together with Partial Dependence to get both "
            "'how much' (importance) and 'which direction' (effect) for the "
            "top fields.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SHAP Global Feature Importance</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 8px 10px; text-align: left; font-size: .9rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
</style></head>
<body><div class="container">
<h1>SHAP Global Feature Importance</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Model: <strong>{html.escape(CV_MODEL_LABELS[model_key])}</strong>{class_note}</p>
<div class="summary">
Explained {n_explained} of {n_total} complete records<br>
Skipped {skipped} record(s) with missing values.
</div>
<h2>Mean |SHAP value| by Field</h2>
<table><thead><tr><th>Field</th><th>Mean |SHAP|</th></tr></thead><tbody>{rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
