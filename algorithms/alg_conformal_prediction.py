# -*- coding: utf-8 -*-
"""Conformal Prediction Interval Processing Algorithm."""
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
from ..core.ml_engines import CV_MODEL_KEYS, CV_MODEL_LABELS, extract_regression_matrix, fit_conformal_interval
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class ConformalPredictionAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    MODEL = "MODEL"
    ALPHA = "ALPHA"
    N_FOLDS = "N_FOLDS"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    MODEL_LABELS = [CV_MODEL_LABELS[key] for key in CV_MODEL_KEYS]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "conformal_prediction_interval"

    def displayName(self) -> str:
        return "Conformal Prediction Interval"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("conformal_prediction_interval")

    def createInstance(self):
        return ConformalPredictionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes distribution-free prediction intervals with a "
            "mathematically guaranteed marginal coverage rate, for any "
            "regression model in this group - not just Random Forest/Extra "
            "Trees the way Prediction Uncertainty Map is limited to. Wraps "
            "the MAPIE library's jackknife+ conformal method.\n\n"
            "Unlike Prediction Uncertainty Map's tree-vote spread (a useful "
            "but heuristic proxy available only for bagging ensembles), "
            "conformal prediction is a statistical procedure: under only the "
            "assumption that the data is exchangeable (a weaker assumption "
            "than the normal-error assumption behind classical confidence "
            "intervals), the resulting interval is proven to cover the true "
            "value at least (1-alpha) of the time in expectation, regardless "
            "of which model produced the point prediction - including "
            "Gradient Boosting, SVM, and MLP, none of which have a natural "
            "uncertainty measure of their own.\n\n"
            "Output: the input layer with conf_pred, conf_low, conf_high, "
            "and conf_width. The report shows empirical in-sample coverage "
            "(the fraction of records whose actual value fell inside its own "
            "interval) alongside the target coverage (1-alpha) - these should "
            "be close, though this in-sample check is optimistic for the "
            "same structural reason every in-sample metric in this group is; "
            "conformal's real coverage guarantee is about future, unseen "
            "predictions under the cross-validated jackknife+ procedure, not "
            "about the number printed for the training data itself.\n\n"
            "Requires the optional mapie package (Setup and Diagnostics > "
            "Install / Update GeoStats Libraries)."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.TARGET, "Target field (numeric)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FEATURES, "Explanatory fields (numeric)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric, allowMultiple=True,
            )
        )
        self.addParameter(QgsProcessingParameterEnum(self.MODEL, "Model", options=self.MODEL_LABELS, defaultValue=0))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ALPHA, "Miscoverage rate (alpha; interval targets 1-alpha coverage)",
                type=QgsProcessingParameterNumber.Double, defaultValue=0.1, minValue=0.01, maxValue=0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_FOLDS, "Cross-conformal folds", type=QgsProcessingParameterNumber.Integer,
                defaultValue=5, minValue=2, maxValue=20,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output layer with prediction intervals"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output conformal prediction HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Conformal prediction report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")
        model_key = CV_MODEL_KEYS[self.parameterAsEnum(parameters, self.MODEL, context)]
        alpha = self.parameterAsDouble(parameters, self.ALPHA, context)
        n_folds = self.parameterAsInt(parameters, self.N_FOLDS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_conformal_prediction_report.html")

        feedback.pushInfo("Extracting complete numeric records...")
        try:
            extraction = extract_regression_matrix(source, feature_fields, target_field, feedback, (0, 20))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, valid_fids, skipped = extraction["x"], extraction["y"], extraction["valid_fids"], extraction["skipped"]
        if len(y) < n_folds * 3:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {n_folds} cross-conformal folds; "
                "reduce Cross-conformal folds or select a layer with more complete records."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric values.")

        feedback.pushInfo(f"Fitting {CV_MODEL_LABELS[model_key]} with jackknife+ conformal calibration...")
        try:
            results = fit_conformal_interval(x, y, model_key, alpha=alpha, cv=n_folds)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Conformal Prediction Interval", ["mapie"], exc))

        feedback.pushInfo(
            f"Target coverage={results['target_coverage']:.3f}, empirical in-sample coverage={results['empirical_coverage']:.3f}"
        )

        out_fields = source.fields()
        out_fields.append(QgsField("conf_pred", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("conf_low", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("conf_high", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("conf_width", QVariant.Double, len=12, prec=6))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        result_map = {fid: idx for idx, fid in enumerate(valid_fids)}
        total = source.featureCount() or 1
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            if fid in result_map:
                row_idx = result_map[fid]
                low = float(results["lower"][row_idx])
                high = float(results["upper"][row_idx])
                out_feature.setAttribute("conf_pred", float(results["pred"][row_idx]))
                out_feature.setAttribute("conf_low", low)
                out_feature.setAttribute("conf_high", high)
                out_feature.setAttribute("conf_width", high - low)
            else:
                out_feature.setAttribute("conf_pred", None)
                out_feature.setAttribute("conf_low", None)
                out_feature.setAttribute("conf_high", None)
                out_feature.setAttribute("conf_width", None)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 70 * (current / total)))

        self._write_html(html_path, target_field, model_key, alpha, n_folds, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats conformal prediction interval output",
            {
                "conf_pred": "Point prediction from the conformal-calibrated model",
                "conf_low": "Lower bound of the (1-alpha) conformal prediction interval",
                "conf_high": "Upper bound of the (1-alpha) conformal prediction interval",
                "conf_width": "Interval width (conf_high - conf_low); wider = less certain",
            },
            "conformal_prediction_interval",
        )
        return {}

    def _write_html(self, path, target_field, model_key, alpha, n_folds, results, skipped):
        mean_width = float(np.mean(results["upper"] - results["lower"]))
        guidance = analyst_guidance_html(
            "Conformal Prediction Interval",
            "Model-agnostic prediction intervals with a distribution-free "
            "marginal coverage guarantee, via cross-conformal jackknife+ "
            "calibration - works for any model, not only bagging ensembles.",
            [
                "Empirical in-sample coverage is reasonably close to the target coverage.",
                "The interval width (conf_width) is examined alongside the point prediction, not ignored.",
            ],
            [
                "conf_width varies enormously across records with no clear pattern (may indicate the model struggles in certain regions - cross-check with SHAP or Prediction Uncertainty Map).",
                "Empirical in-sample coverage is far below target (the exchangeability assumption may be violated, e.g., by strong spatial autocorrelation in the residuals - check Model Residual Spatial Autocorrelation Check).",
            ],
            [
                "Model Residual Spatial Autocorrelation Check - confirm the exchangeability assumption is not badly violated by spatial structure.",
                "Prediction Uncertainty Map - a complementary, tree-specific uncertainty view for Random Forest/Extra Trees.",
            ],
            "Report the interval [conf_low, conf_high], not just conf_pred, "
            "when a prediction will inform a planning decision - the interval "
            "width is itself information about how much to trust that specific record's prediction.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Conformal Prediction Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
</style></head>
<body><div class="container">
<h1>Conformal Prediction Interval</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Model: <strong>{html.escape(CV_MODEL_LABELS[model_key])}</strong> | Alpha: <strong>{alpha:g}</strong> | Folds: <strong>{n_folds}</strong></p>
<div class="summary">
Target coverage = {results['target_coverage']:.4f} | Empirical in-sample coverage = {results['empirical_coverage']:.4f} | Mean interval width = {mean_width:.6f}<br>
Skipped {skipped} record(s) with missing or non-numeric values.
</div>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
