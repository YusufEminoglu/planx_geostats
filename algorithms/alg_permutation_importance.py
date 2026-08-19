# -*- coding: utf-8 -*-
"""Permutation Feature Importance Processing Algorithm."""
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
    permutation_feature_importance,
)
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, bar_chart_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class PermutationImportanceAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    TASK_TYPE = "TASK_TYPE"
    MODEL = "MODEL"
    N_REPEATS = "N_REPEATS"
    HTML_REPORT = "HTML_REPORT"

    TASK_TYPES = ["Regression (numeric target)", "Classification (categorical target)"]
    TASK_KEYS = ["regression", "classification"]
    MODEL_LABELS = [CV_MODEL_LABELS[key] for key in CV_MODEL_KEYS]

    def name(self) -> str:
        return "permutation_feature_importance"

    def displayName(self) -> str:
        return "Permutation Feature Importance"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("permutation_feature_importance")

    def createInstance(self):
        return PermutationImportanceAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Ranks explanatory fields by how much a fitted model's score drops "
            "when that field's values are randomly shuffled (breaking its "
            "relationship with the target) while every other field stays "
            "intact. A field whose shuffling barely changes the score "
            "contributed little; a field whose shuffling collapses the score "
            "was doing most of the work.\n\n"
            "Unlike tree-based feature_importances_ (used by Random Forest and "
            "Extra Trees), this method is model-agnostic and works for every "
            "model in this group, including Support Vector Machine, Neural "
            "Network, and the scikit-learn Gradient Boosting engine, none of "
            "which expose a native importance ranking.\n\n"
            "Output: an HTML report with each field's mean importance and its "
            "variability across repeated shuffles (a wide range across repeats "
            "means the estimate itself is noisy - raise Number of repeats). The "
            "baseline score (R2 or accuracy on the same data used to fit) is "
            "shown for reference; this tool does not cross-validate, so pair it "
            "with Spatial k-Fold Cross-Validation Evaluator when you need an "
            "honest accuracy figure alongside the importance ranking.\n\n"
            "A near-zero or negative importance for a field means the model did "
            "not rely on it - consider dropping it in a follow-up run, both to "
            "simplify the model and to check whether the remaining fields' "
            "ranking is stable without it."
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
            QgsProcessingParameterNumber(
                self.N_REPEATS, "Number of shuffle repeats", type=QgsProcessingParameterNumber.Integer,
                defaultValue=10, minValue=3, maxValue=100,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output permutation importance HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Permutation importance report"))

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
        n_repeats = self.parameterAsInt(parameters, self.N_REPEATS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_permutation_importance_report.html")

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

        feedback.pushInfo(f"Fitting {CV_MODEL_LABELS[model_key]} and computing permutation importance...")
        try:
            results = permutation_feature_importance(x, y, feature_fields, task_type, model_key, n_repeats=n_repeats)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Permutation Feature Importance", ["scikit-learn"], exc))

        feedback.pushInfo(f"Baseline {results['scoring']} = {results['baseline_score']:.4f}")

        self._write_html(html_path, target_field, task_type, model_key, n_repeats, results, skipped)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, target_field, task_type, model_key, n_repeats, results, skipped):
        rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{mean:.6f}</td><td>{std:.6f}</td></tr>"
            for name, mean, std in results["rows"]
        )
        importance_chart = bar_chart_svg(
            [name for name, _mean, _std in results["rows"]],
            [mean for _name, mean, _std in results["rows"]],
            title="",
            value_suffix=f" {results['scoring']} drop",
        )
        guidance = analyst_guidance_html(
            "Permutation Feature Importance",
            "Ranks fields by how much a fitted model's score drops when that "
            "field is randomly shuffled; works for any model, including ones "
            "with no native importance measure.",
            [
                "Importance std (across repeats) is small relative to the mean, so the ranking is stable.",
                "The baseline score is reasonable (this method's importances are only meaningful for a model that fits reasonably well).",
            ],
            [
                "Every field's importance is near zero (the model may not be learning a real relationship).",
                "Wide variability across repeats (raise Number of repeats for a more stable estimate).",
            ],
            [
                "Spatial k-Fold Cross-Validation Evaluator - confirm the baseline score is not overfit to this sample.",
                "SHAP Global Feature Importance - a complementary, game-theoretic ranking (tree models only get a fast exact version).",
            ],
            "Use the ranking to decide which fields are worth keeping in a "
            "simplified model, and to sanity-check that the model relies on "
            "planning-relevant fields rather than an accidental proxy.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Permutation Feature Importance Report</title>
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
<h1>Permutation Feature Importance</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Model: <strong>{html.escape(CV_MODEL_LABELS[model_key])}</strong> | Repeats: <strong>{n_repeats}</strong></p>
<div class="summary">
Baseline {html.escape(results['scoring'])} = {results['baseline_score']:.6f}<br>
Skipped {skipped} record(s) with missing values.
</div>
<h2>Importance by Field (mean score drop &plusmn; std across repeats)</h2>
{importance_chart}
<table><thead><tr><th>Field</th><th>Mean importance</th><th>Std</th></tr></thead><tbody>{rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
