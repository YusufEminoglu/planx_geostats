# -*- coding: utf-8 -*-
"""Spatial k-Fold Cross-Validation Evaluator Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.ml_engines import CV_MODEL_KEYS, CV_MODEL_LABELS, extract_matrix_with_centroids, spatial_kfold_evaluate
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, bar_chart_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class SpatialCVEvaluatorAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    TASK_TYPE = "TASK_TYPE"
    MODEL = "MODEL"
    N_FOLDS = "N_FOLDS"
    FOLD_METHOD = "FOLD_METHOD"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    TASK_TYPES = ["Regression (numeric target)", "Classification (categorical target)"]
    TASK_KEYS = ["regression", "classification"]
    MODEL_LABELS = [CV_MODEL_LABELS[key] for key in CV_MODEL_KEYS]
    FOLD_METHODS = ["K-Means spatial block (default)", "kNNDM (nearest-neighbour distance matching)"]
    FOLD_METHOD_KEYS = ["kmeans_block", "knndm"]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "spatial_kfold_cv_evaluator"

    def displayName(self) -> str:
        return "Spatial k-Fold Cross-Validation Evaluator"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("spatial_kfold_cv_evaluator")

    def createInstance(self):
        return SpatialCVEvaluatorAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Estimates honest out-of-sample accuracy for any of the tree, "
            "kernel, or neural-network models in this group by k-fold cross-"
            "validation with spatially contiguous folds instead of randomly "
            "shuffled ones. Records are grouped into k geographic blocks (via "
            "K-Means on feature centroid coordinates), then each block is held "
            "out in turn while the model trains on the rest.\n\n"
            "This matters because planning data is almost always spatially "
            "autocorrelated: two neighboring parcels tend to be more similar "
            "than two distant ones. Ordinary random k-fold CV lets train and "
            "test points from the same neighborhood leak information across "
            "the split, which inflates the reported score - the in-sample R2 or "
            "accuracy shown by every other tool in this group has exactly this "
            "optimism problem. Spatial blocking removes that leakage and "
            "produces a more realistic estimate of how the model will perform "
            "on genuinely new areas.\n\n"
            "Output: the input layer with a cv_fold field (which spatial block "
            "each record fell into - map it to see the block layout), plus an "
            "HTML report with per-fold metrics and their mean and standard "
            "deviation across folds. A large standard deviation across folds "
            "means performance is uneven across the study area - check whether "
            "one region has systematically different data quality or a "
            "different underlying process.\n\n"
            "Choose Number of folds based on data size: 5 is a reasonable "
            "default; fewer folds (3) for smaller datasets, more (10) only with "
            "several hundred records or more so each held-out block still has "
            "enough test records to compute a stable metric.\n\n"
            "Fold method: K-Means spatial block (the default above) groups "
            "records purely by geographic compactness. kNNDM (Linnenbrink, "
            "Milà, Ludwig and Meyer, Geoscientific Model Development, 2024) "
            "instead searches for the fold assignment whose test-to-nearest-"
            "training-point distance distribution most closely matches the "
            "dataset's own leave-one-out nearest-neighbour distance "
            "distribution - i.e. the split that makes cross-validation "
            "distances 'feel' most like predicting at a genuinely new "
            "location, which K-Means blocking does not explicitly optimize "
            "for. kNNDM costs more compute (it searches many candidate fold "
            "assignments) and can produce less visually tidy blocks; prefer "
            "it when the honesty of the reported score matters more than a "
            "clean block map, and prefer K-Means spatial block when you also "
            "want to inspect cv_fold as a simple regional breakdown."
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
                self.N_FOLDS, "Number of spatial folds (k)", type=QgsProcessingParameterNumber.Integer,
                defaultValue=5, minValue=2, maxValue=20,
            )
        )
        self.addParameter(QgsProcessingParameterEnum(self.FOLD_METHOD, "Fold method", options=self.FOLD_METHODS, defaultValue=0))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output layer with fold assignment"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output cross-validation HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Spatial cross-validation report"))

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
        n_folds = self.parameterAsInt(parameters, self.N_FOLDS, context)
        fold_method = self.FOLD_METHOD_KEYS[self.parameterAsEnum(parameters, self.FOLD_METHOD, context)]

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_spatial_cv_report.html")

        feedback.pushInfo("Extracting complete records and centroids...")
        try:
            extraction = extract_matrix_with_centroids(source, feature_fields, target_field, task_type, feedback, (0, 20))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, centroids = extraction["x"], extraction["y"], extraction["centroids"]
        valid_fids, skipped = extraction["valid_fids"], extraction["skipped"]
        class_labels = extraction.get("class_labels")
        if task_type == "classification" and (not class_labels or len(class_labels) < 2):
            raise QgsProcessingException("At least 2 distinct classes are required for classification cross-validation.")
        if len(y) < n_folds * 2:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {n_folds} spatial folds; "
                "reduce Number of folds or select a layer with more complete records."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing values or empty geometry.")

        method_label = self.FOLD_METHODS[self.FOLD_METHOD_KEYS.index(fold_method)]
        feedback.pushInfo(f"Running spatial {n_folds}-fold CV ({method_label}) with {CV_MODEL_LABELS[model_key]}...")
        try:
            cv_results = spatial_kfold_evaluate(x, y, centroids, n_folds, task_type, model_key, fold_method=fold_method)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Spatial k-Fold Cross-Validation Evaluator", ["scikit-learn"], exc))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        fold_metrics = cv_results["fold_metrics"]
        if not fold_metrics:
            raise QgsProcessingException("No fold produced both training and test records; try fewer folds.")

        metric_key = "r2" if task_type == "regression" else "accuracy"
        values = [row[metric_key] for row in fold_metrics]
        feedback.pushInfo(
            f"Mean {metric_key} across {len(fold_metrics)} folds: {np.mean(values):.4f} "
            f"(std {np.std(values):.4f})"
        )

        out_fields = source.fields()
        out_fields.append(QgsField("cv_fold", QVariant.Int))
        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        fold_map = {fid: int(fold) for fid, fold in zip(valid_fids, cv_results["folds"])}
        total = source.featureCount() or 1
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            out_feature.setAttribute("cv_fold", fold_map.get(fid))
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, task_type, model_key, n_folds, method_label, fold_metrics, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats spatial cross-validation fold assignment output",
            {"cv_fold": "Spatial block (fold) index this record was assigned to (NULL if it had missing values)"},
            "spatial_kfold_cv_evaluator",
        )
        return {}

    def _write_html(self, path, target_field, feature_fields, task_type, model_key, n_folds, method_label, fold_metrics, skipped):
        is_regression = task_type == "regression"
        metric_key = "r2" if is_regression else "accuracy"
        headers = ["Fold", "Train n", "Test n"] + (["R2", "RMSE", "MAE"] if is_regression else ["Accuracy", "Macro F1"])
        rows = []
        for row in fold_metrics:
            cells = [str(row["fold"]), str(row["n_train"]), str(row["n_test"])]
            if is_regression:
                cells += [f"{row['r2']:.4f}", f"{row['rmse']:.4f}", f"{row['mae']:.4f}"]
            else:
                cells += [f"{row['accuracy']:.4f}", f"{row['macro_f1']:.4f}"]
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        values = np.array([row[metric_key] for row in fold_metrics])
        fold_chart = bar_chart_svg(
            [f"Fold {row['fold']}" for row in fold_metrics],
            values.tolist(),
            horizontal=False,
        )
        guidance = analyst_guidance_html(
            "Spatial k-Fold Cross-Validation",
            "Held-out estimate of model accuracy using geographically "
            "contiguous folds, so nearby train/test records cannot leak "
            "information across the split.",
            [
                "Standard deviation across folds is modest relative to the mean.",
                "The mean metric here is noticeably lower than the fitting tool's in-sample score (expected; a large gap is normal, not a bug).",
            ],
            [
                "One fold's score is far below the others (a spatially distinct subregion the model handles poorly).",
                "Mean score barely above a naive baseline (mean-predictor for regression, majority-class for classification).",
            ],
            [
                "Map cv_fold to see whether the low-scoring fold corresponds to a specific district or data-quality issue.",
                "SHAP Global Feature Importance - check whether the same fields matter everywhere or only in some folds.",
            ],
            "Report this cross-validated score, not the fitting tool's in-sample "
            "score, as your model's real-world accuracy estimate.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Spatial Cross-Validation Report</title>
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
<h1>Spatial k-Fold Cross-Validation</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Model: <strong>{html.escape(CV_MODEL_LABELS[model_key])}</strong> | Folds: <strong>{n_folds}</strong> | Fold method: <strong>{html.escape(method_label)}</strong></p>
<div class="summary">
Mean {metric_key} = {float(np.mean(values)):.6f} | Std across folds = {float(np.std(values)):.6f}<br>
Skipped {skipped} record(s) with missing values or empty geometry.
</div>
<h2>Per-Fold Metrics</h2>
{fold_chart}
<table><thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead><tbody>{''.join(rows)}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
