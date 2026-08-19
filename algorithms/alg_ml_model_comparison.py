# -*- coding: utf-8 -*-
"""ML Model Comparison / Leaderboard Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

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

from ..core.ml_engines import CV_MODEL_KEYS, CV_MODEL_LABELS, extract_matrix_with_centroids, spatial_kfold_evaluate
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, bar_chart_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class MLModelComparisonAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    TASK_TYPE = "TASK_TYPE"
    MODELS = "MODELS"
    N_FOLDS = "N_FOLDS"
    HTML_REPORT = "HTML_REPORT"

    TASK_TYPES = ["Regression (numeric target)", "Classification (categorical target)"]
    TASK_KEYS = ["regression", "classification"]
    MODEL_LABELS = [CV_MODEL_LABELS[key] for key in CV_MODEL_KEYS]

    def name(self) -> str:
        return "ml_model_comparison"

    def displayName(self) -> str:
        return "ML Model Comparison (Leaderboard)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("ml_model_comparison")

    def createInstance(self):
        return MLModelComparisonAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Runs Spatial k-Fold Cross-Validation for every selected model type "
            "(Random Forest, Extra Trees, Gradient Boosting, Support Vector "
            "Machine, Neural Network) on the same target and explanatory "
            "fields, then ranks them by mean out-of-sample score in a single "
            "leaderboard - so you compare models on an honest, apples-to-apples "
            "basis instead of eyeballing separate reports from each tool.\n\n"
            "Output: an HTML report ranking every evaluated model by mean R2 "
            "(regression) or accuracy (classification) across spatial folds, "
            "with the standard deviation across folds shown alongside so a "
            "close leaderboard finish is not mistaken for a clear winner.\n\n"
            "This tool only tests the scikit-learn Gradient Boosting engine "
            "(no xgboost/lightgbm dependency required to run a comparison); "
            "once you have picked a winning model family here, use the "
            "matching dedicated tool - Gradient Boosting Regression/"
            "Classification also offers the XGBoost and LightGBM engines if "
            "boosting wins.\n\n"
            "A leaderboard where every model scores within one standard "
            "deviation of each other means model choice barely matters for "
            "this dataset - in that case, prefer the simplest or most "
            "interpretable option (Random Forest) over a marginally higher-"
            "ranked but harder-to-explain one."
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
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODELS, "Models to compare", options=self.MODEL_LABELS,
                allowMultiple=True, defaultValue=list(range(len(self.MODEL_LABELS))),
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_FOLDS, "Number of spatial folds (k)", type=QgsProcessingParameterNumber.Integer,
                defaultValue=5, minValue=2, maxValue=20,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output leaderboard HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "ML model comparison leaderboard"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        task_type = self.TASK_KEYS[self.parameterAsEnum(parameters, self.TASK_TYPE, context)]
        model_indices = self.parameterAsEnums(parameters, self.MODELS, context)
        if not model_indices:
            raise QgsProcessingException("At least one model must be selected.")
        selected_models = [CV_MODEL_KEYS[i] for i in model_indices]
        n_folds = self.parameterAsInt(parameters, self.N_FOLDS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_ml_model_comparison_report.html")

        feedback.pushInfo("Extracting complete records and centroids...")
        try:
            extraction = extract_matrix_with_centroids(source, feature_fields, target_field, task_type, feedback, (0, 15))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, centroids = extraction["x"], extraction["y"], extraction["centroids"]
        skipped = extraction["skipped"]
        if task_type == "classification" and len(extraction.get("class_labels", [])) < 2:
            raise QgsProcessingException("At least 2 distinct classes are required for classification.")
        if len(y) < n_folds * 2:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {n_folds} spatial folds."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing values or empty geometry.")

        metric_key = "r2" if task_type == "regression" else "accuracy"
        leaderboard = []
        for step, model_key in enumerate(selected_models):
            feedback.pushInfo(f"Evaluating {CV_MODEL_LABELS[model_key]} ({step + 1}/{len(selected_models)})...")
            try:
                cv_results = spatial_kfold_evaluate(x, y, centroids, n_folds, task_type, model_key)
            except ImportError as exc:
                feedback.pushWarning(
                    optional_dependency_error(f"ML Model Comparison ({CV_MODEL_LABELS[model_key]})", ["scikit-learn"], exc)
                )
                continue
            fold_metrics = cv_results["fold_metrics"]
            if not fold_metrics:
                feedback.pushWarning(f"{CV_MODEL_LABELS[model_key]}: no fold produced both train and test records, skipped.")
                continue
            values = np.array([row[metric_key] for row in fold_metrics])
            leaderboard.append({
                "model": model_key,
                "label": CV_MODEL_LABELS[model_key],
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_folds": len(fold_metrics),
            })
            feedback.setProgress(int(15 + 80 * ((step + 1) / len(selected_models))))

        if not leaderboard:
            raise QgsProcessingException("No model could be evaluated; check the optional-dependency warnings above.")

        leaderboard.sort(key=lambda row: row["mean"], reverse=True)
        feedback.pushInfo(
            f"Best model: {leaderboard[0]['label']} (mean {metric_key}={leaderboard[0]['mean']:.4f}, "
            f"std={leaderboard[0]['std']:.4f})"
        )

        self._write_html(html_path, target_field, task_type, n_folds, leaderboard, skipped)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, target_field, task_type, n_folds, leaderboard, skipped):
        metric_label = "Mean R2" if task_type == "regression" else "Mean Accuracy"
        rows = "".join(
            f"<tr><td>{i + 1}</td><td>{html.escape(row['label'])}</td>"
            f"<td>{row['mean']:.6f}</td><td>{row['std']:.6f}</td><td>{row['n_folds']}</td></tr>"
            for i, row in enumerate(leaderboard)
        )
        top = leaderboard[0]
        close_finish = len(leaderboard) > 1 and (top["mean"] - leaderboard[1]["mean"]) < top["std"]
        verdict = (
            "The top two models finish within one standard deviation of each "
            "other - treat this as a near-tie and prefer the simpler, more "
            "interpretable model rather than the nominal leader."
            if close_finish else
            f"{html.escape(top['label'])} leads by more than one standard "
            "deviation over the runner-up - a reasonably clear result for this "
            "dataset and field selection."
        )
        leaderboard_chart = bar_chart_svg(
            [row["label"] for row in leaderboard],
            [row["mean"] for row in leaderboard],
            value_suffix=f" {metric_label}",
            highlight_index=0,
        )
        guidance = analyst_guidance_html(
            "ML Model Comparison",
            "Ranks candidate model families by mean spatially cross-validated "
            "score on the same fields, so the comparison is apples-to-apples.",
            [
                "Every model in the leaderboard reached at least one successful fold.",
                "The winning model's lead over the runner-up exceeds its cross-fold standard deviation.",
            ],
            [
                "Every model scores within one standard deviation of each other (model choice barely matters here).",
                "A model dropped out due to a missing optional dependency (check the Processing log warnings).",
            ],
            [
                "Run the dedicated tool for the winning model family to get its full diagnostic report.",
                "SHAP Global Feature Importance - explain the winning model's predictions.",
            ],
            verdict,
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ML Model Comparison Leaderboard</title>
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
<h1>ML Model Comparison Leaderboard</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Folds: <strong>{n_folds}</strong></p>
<div class="summary">
Best model: <strong>{html.escape(top['label'])}</strong> ({metric_label} = {top['mean']:.6f} &plusmn; {top['std']:.6f})<br>
Skipped {skipped} record(s) with missing values or empty geometry.
</div>
<h2>Leaderboard</h2>
{leaderboard_chart}
<table><thead><tr><th>Rank</th><th>Model</th><th>{metric_label}</th><th>Std</th><th>Folds used</th></tr></thead><tbody>{rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
