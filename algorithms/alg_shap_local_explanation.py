# -*- coding: utf-8 -*-
"""SHAP Local Explanation Report Processing Algorithm."""
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
from ..core.xai_engines import compute_shap_values, shap_local_breakdown
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class SHAPLocalExplanationAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    TASK_TYPE = "TASK_TYPE"
    MODEL = "MODEL"
    CLASS_LABEL = "CLASS_LABEL"
    FEATURE_ID = "FEATURE_ID"
    HTML_REPORT = "HTML_REPORT"

    TASK_TYPES = ["Regression (numeric target)", "Classification (categorical target)"]
    TASK_KEYS = ["regression", "classification"]
    MODEL_LABELS = [CV_MODEL_LABELS[key] for key in CV_MODEL_KEYS]

    def name(self) -> str:
        return "shap_local_explanation"

    def displayName(self) -> str:
        return "SHAP Local Explanation Report"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("shap_local_explanation")

    def createInstance(self):
        return SHAPLocalExplanationAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits the chosen model on the full layer, then computes a full "
            "SHAP breakdown for exactly one record (identified by its QGIS "
            "feature ID, visible in the attribute table or via Identify Features "
            "on the map): every explanatory field's contribution, signed and "
            "ranked by magnitude, that sums with the base value to that "
            "record's prediction.\n\n"
            "Use this after SHAP Spatial Attribution Map or the Identify tool "
            "points you to a specific feature worth understanding - an "
            "outlier prediction, a surprising high-confidence classification, "
            "or a record a stakeholder is asking about specifically.\n\n"
            "Output: an HTML report listing every field's contribution for "
            "that one record, ordered by absolute size, alongside the base "
            "value (average prediction) and the final predicted value they "
            "sum to.\n\n"
            "For classification, the breakdown is for one class at a time - "
            "set Class to explain, or leave blank for the first class in "
            "sorted label order. A positive contribution pushes the "
            "prediction toward that class/higher value; negative pushes away "
            "from it/lower."
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
                self.FEATURE_ID, "Feature ID to explain", type=QgsProcessingParameterNumber.Integer,
                defaultValue=0, minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output SHAP local explanation HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "SHAP local explanation report"))

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
        feature_id = self.parameterAsInt(parameters, self.FEATURE_ID, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_shap_local_report.html")

        feedback.pushInfo("Extracting complete records...")
        try:
            if task_type == "regression":
                extraction = extract_regression_matrix(source, feature_fields, target_field, feedback, (0, 20))
            else:
                extraction = extract_classification_matrix(source, feature_fields, target_field, feedback, (0, 20))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, valid_fids, skipped = extraction["x"], extraction["y"], extraction["valid_fids"], extraction["skipped"]
        if len(y) <= len(feature_fields) + 1:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {len(feature_fields)} explanatory field(s)."
            )
        if feature_id not in valid_fids:
            raise QgsProcessingException(
                f"Feature ID {feature_id} was not among the complete records used to fit the model "
                "(it may not exist, or it may have missing values in the target/explanatory fields)."
            )
        row_index = valid_fids.index(feature_id)

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

        feedback.pushInfo(f"Fitting {CV_MODEL_LABELS[model_key]} and computing SHAP values for feature {feature_id}...")
        try:
            shap_result = compute_shap_values(
                x, y, task_type, model_key, class_index=class_index, explicit_rows=[row_index],
            )
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("SHAP Local Explanation Report", ["shap"], exc))

        breakdown = shap_local_breakdown(shap_result["shap_values"], feature_fields, 0, shap_result["base_value"])
        actual_target = y[row_index]
        explained_class = class_labels[class_index] if task_type == "classification" else None

        self._write_html(html_path, feature_id, target_field, model_key, explained_class, actual_target, breakdown)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, feature_id, target_field, model_key, explained_class, actual_target, breakdown):
        rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{value:+.6f}</td></tr>"
            for name, value in breakdown["contributions"]
        )
        class_note = f" | Class explained: <strong>{html.escape(explained_class)}</strong>" if explained_class else ""
        actual_note = "" if explained_class else f" | Actual target value: <strong>{actual_target:.6f}</strong>"
        guidance = analyst_guidance_html(
            "SHAP Local Explanation",
            "Signed, ranked per-field contributions for one specific record, "
            "summing exactly with the base value to that record's prediction.",
            [
                "The top contributing fields make planning sense for this specific location.",
                "The predicted value is reasonably close to the actual target value (a large gap means the model is uncertain about exactly this record).",
            ],
            [
                "The prediction is driven mostly by one field with no obvious planning explanation (check for a data-entry error in that field for this record).",
                "Predicted value is far from the actual target value (the model struggles with this specific record).",
            ],
            [
                "SHAP Spatial Attribution Map - see whether this record's top field is also important nearby, or if it is a local exception.",
                "Partial Dependence Report - see the average effect of the top contributing field across all records.",
            ],
            "Read positive contributions as pushing the prediction up (or "
            "toward the explained class) and negative as pushing it down (or "
            "away from it); the ranking is by absolute size, not sign.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SHAP Local Explanation Report</title>
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
<h1>SHAP Local Explanation - Feature ID {feature_id}</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Model: <strong>{html.escape(CV_MODEL_LABELS[model_key])}</strong>{class_note}{actual_note}</p>
<div class="summary">
Base value = {breakdown['base_value']:.6f} | Predicted value = {breakdown['prediction']:.6f}
</div>
<h2>Field Contributions (ranked by |SHAP value|)</h2>
<table><thead><tr><th>Field</th><th>SHAP contribution</th></tr></thead><tbody>{rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
