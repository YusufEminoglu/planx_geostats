# -*- coding: utf-8 -*-
"""Partial Dependence Report Processing Algorithm."""
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
)

from ..core.ml_engines import (
    CV_MODEL_KEYS,
    CV_MODEL_LABELS,
    extract_classification_matrix,
    extract_regression_matrix,
    partial_dependence_report,
)
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, line_chart_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class PartialDependenceAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    PD_FIELD = "PD_FIELD"
    TASK_TYPE = "TASK_TYPE"
    MODEL = "MODEL"
    GRID_POINTS = "GRID_POINTS"
    HTML_REPORT = "HTML_REPORT"

    TASK_TYPES = ["Regression (numeric target)", "Classification (categorical target)"]
    TASK_KEYS = ["regression", "classification"]
    MODEL_LABELS = [CV_MODEL_LABELS[key] for key in CV_MODEL_KEYS]

    def name(self) -> str:
        return "partial_dependence_report"

    def displayName(self) -> str:
        return "Partial Dependence Report"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("partial_dependence_report")

    def createInstance(self):
        return PartialDependenceAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a model on the selected explanatory fields, then shows how "
            "the average prediction changes as one chosen field is swept "
            "across its observed range while every other field is held at "
            "each record's actual value and the results averaged (Friedman's "
            "Partial Dependence Plot). This answers 'holding everything else "
            "fixed, what does increasing this field do to the prediction' - "
            "the direction-of-effect question that feature importances and "
            "SHAP summary tools do not answer on their own.\n\n"
            "Output: an HTML report with a curve table of field-value versus "
            "average predicted outcome (for classification, the predicted "
            "probability of the first class in sorted class-label order).\n\n"
            "Partial dependence assumes the swept field is independent of the "
            "other explanatory fields; if it is strongly correlated with "
            "another selected field (for example two different density "
            "measures), the curve can be misleading because the combinations "
            "it evaluates may not occur in reality. Check correlations between "
            "explanatory fields first, or use SHAP Local Explanation Report on "
            "individual records for an alternative that does not require this "
            "assumption as strongly.\n\n"
            "A flat curve means the model's predictions barely change with "
            "this field once the model has been fit - not necessarily that the "
            "field is unrelated to the outcome on its own, only that it adds "
            "little once the other selected fields are already in the model."
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
                self.FEATURES, "Explanatory fields (numeric, model context)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric, allowMultiple=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.PD_FIELD, "Field to compute partial dependence for (must be one of the explanatory fields)",
                parentLayerParameterName=self.INPUT, type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(QgsProcessingParameterEnum(self.TASK_TYPE, "Task type", options=self.TASK_TYPES, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.MODEL, "Model", options=self.MODEL_LABELS, defaultValue=0))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.GRID_POINTS, "Grid points", type=QgsProcessingParameterNumber.Integer,
                defaultValue=20, minValue=5, maxValue=200,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output partial dependence HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Partial dependence report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        pd_field = self.parameterAsString(parameters, self.PD_FIELD, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")
        if pd_field not in feature_fields:
            raise QgsProcessingException("The partial dependence field must be one of the selected explanatory fields.")

        task_type = self.TASK_KEYS[self.parameterAsEnum(parameters, self.TASK_TYPE, context)]
        model_key = CV_MODEL_KEYS[self.parameterAsEnum(parameters, self.MODEL, context)]
        grid_points = self.parameterAsInt(parameters, self.GRID_POINTS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_partial_dependence_report.html")

        feedback.pushInfo("Extracting complete records...")
        try:
            if task_type == "regression":
                extraction = extract_regression_matrix(source, feature_fields, target_field, feedback, (0, 30))
            else:
                extraction = extract_classification_matrix(source, feature_fields, target_field, feedback, (0, 30))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, skipped = extraction["x"], extraction["y"], extraction["skipped"]
        if len(y) <= len(feature_fields) + 1:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {len(feature_fields)} explanatory field(s)."
            )
        if task_type == "classification" and len(extraction["class_labels"]) < 2:
            raise QgsProcessingException("At least 2 distinct classes are required for classification.")
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing values.")

        feature_index = feature_fields.index(pd_field)
        feedback.pushInfo(f"Fitting {CV_MODEL_LABELS[model_key]} and computing partial dependence for '{pd_field}'...")
        try:
            results = partial_dependence_report(x, y, feature_index, task_type, model_key, grid_resolution=grid_points)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Partial Dependence Report", ["scikit-learn"], exc))

        class_note = ""
        if task_type == "classification":
            class_note = f" (probability of class '{extraction['class_labels'][0]}')"

        self._write_html(html_path, target_field, pd_field, model_key, class_note, results, skipped)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, target_field, pd_field, model_key, class_note, results, skipped):
        rows = "".join(
            f"<tr><td>{value:.6f}</td><td>{pred:.6f}</td></tr>"
            for value, pred in zip(results["grid_values"], results["average"])
        )
        pd_chart = line_chart_svg(
            results["grid_values"],
            {"Average prediction": results["average"]},
            x_label=pd_field,
            y_label=f"Average prediction{class_note}",
        )
        guidance = analyst_guidance_html(
            "Partial Dependence",
            "Average model prediction as one field is swept across its "
            "observed range with every other field held at each record's "
            "actual values - the direction-of-effect view SHAP summaries and "
            "feature importances do not provide directly.",
            [
                "The swept field is not strongly correlated with the other explanatory fields.",
                "The underlying model fits reasonably well (a poorly fit model's PDP is not trustworthy).",
            ],
            [
                "A completely flat curve across the whole range (the field may add nothing once the others are in the model).",
                "Strong correlation between the swept field and another explanatory field (check before trusting the curve's shape).",
            ],
            [
                "SHAP Spatial Attribution Map - see where this field's effect is strongest geographically.",
                "SHAP Local Explanation Report - effect for one specific record instead of an average.",
            ],
            "Read the curve's shape (increasing, decreasing, U-shaped, "
            "plateauing) as the field's marginal relationship with the "
            "outcome, not as a causal claim.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Partial Dependence Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 8px 10px; text-align: left; font-size: .9rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
{chart_css()}
</style></head>
<body><div class="container">
<h1>Partial Dependence</h1>
<p>Target: <strong>{html.escape(target_field)}</strong>{html.escape(class_note)} | Field: <strong>{html.escape(pd_field)}</strong> | Model: <strong>{html.escape(CV_MODEL_LABELS[model_key])}</strong></p>
<div class="summary">Skipped {skipped} record(s) with missing values.</div>
<h2>Partial Dependence Curve</h2>
{pd_chart}
<table><thead><tr><th>{html.escape(pd_field)} value</th><th>Average prediction{html.escape(class_note)}</th></tr></thead><tbody>{rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
