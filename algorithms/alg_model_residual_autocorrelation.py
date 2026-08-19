# -*- coding: utf-8 -*-
"""Model Residual Spatial Autocorrelation Check Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

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
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

from ..core.analysis_diagnostics import (
    filter_weights_to_valid_ids,
    push_residual_spatial_diagnostics,
    residual_spatial_autocorrelation_html,
    residual_spatial_autocorrelation_summary,
)
from ..core.layer_metadata import apply_output_metadata
from ..core.symbology import apply_renderer, diverging_residual_renderer
from ..core.ml_engines import CV_MODEL_KEYS, CV_MODEL_LABELS, build_cv_estimator, extract_regression_matrix
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, scatter_plot_svg
from ..core.weights import build_weights_matrix
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class ModelResidualAutocorrelationAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    MODEL = "MODEL"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    MODEL_LABELS = [CV_MODEL_LABELS[key] for key in CV_MODEL_KEYS]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "model_residual_autocorrelation"

    def displayName(self) -> str:
        return "Model Residual Spatial Autocorrelation Check"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("model_residual_autocorrelation")

    def createInstance(self):
        return ModelResidualAutocorrelationAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits the chosen regression model, then runs the same Global "
            "Moran's I test that Generalized Linear Regression, Spatial Lag, "
            "and Spatial Error Regression already run on their residuals - "
            "applied here to a machine-learning model's residuals instead of "
            "a linear one. A significant positive Moran's I on the residuals "
            "means the model has left spatially structured error unexplained: "
            "some geographic factor (proximity to a boundary, a district "
            "effect, a spatially varying process the explanatory fields do "
            "not capture) is still driving part of the outcome.\n\n"
            "This matters for machine-learning models specifically because "
            "their strong in-sample fit can hide the problem - a Random "
            "Forest or Gradient Boosting model can reach a high R2 by fitting "
            "noise in features while still leaving a real spatial signal in "
            "the residuals if none of the explanatory fields carry location "
            "information (distance-to-center, adjacency, or coordinates "
            "themselves).\n\n"
            "Output: the input layer with a model_resid field (map it to see "
            "the pattern directly), plus an HTML report with Moran's I, "
            "z-score, p-value, and a plain-language status.\n\n"
            "If residuals are significantly clustered, consider adding a "
            "spatial-lag or distance-to-center field to the explanatory set "
            "and refitting, or switching to a model in the Models and "
            "Scenarios group (GWR, MGWR, Spatial Lag/Error Regression) built "
            "specifically to account for spatial structure."
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
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output layer with model residuals"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output residual autocorrelation HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Model residual spatial autocorrelation report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")
        model_key = CV_MODEL_KEYS[self.parameterAsEnum(parameters, self.MODEL, context)]

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_model_residual_autocorrelation_report.html")

        feedback.pushInfo("Extracting complete numeric records...")
        try:
            extraction = extract_regression_matrix(source, feature_fields, target_field, feedback, (0, 20))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, valid_fids, skipped = extraction["x"], extraction["y"], extraction["valid_fids"], extraction["skipped"]
        if len(y) <= len(feature_fields) + 1:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {len(feature_fields)} explanatory field(s)."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric values.")

        feedback.pushInfo(f"Fitting {CV_MODEL_LABELS[model_key]}...")
        try:
            estimator = build_cv_estimator("regression", model_key)
            estimator.fit(x, y)
        except ImportError as exc:
            raise QgsProcessingException(
                optional_dependency_error("Model Residual Spatial Autocorrelation Check", ["scikit-learn"], exc)
            )
        fitted = estimator.predict(x)
        residuals = y - fitted

        weight_type = "queen" if QgsWkbTypes.geometryType(source.wkbType()) == QgsWkbTypes.PolygonGeometry else "knn"
        feedback.pushInfo(f"Building {weight_type} weights for residual spatial autocorrelation diagnostics...")
        try:
            neighbors, _, _, _ = build_weights_matrix(source, weight_type, k_neighbors=8, feedback=feedback)
            filtered_neighbors, filtered_weights, filtered_ids = filter_weights_to_valid_ids(neighbors, valid_fids)
            summary = residual_spatial_autocorrelation_summary(residuals, filtered_neighbors, filtered_weights, filtered_ids)
        except Exception as exc:
            summary = {
                "available": False, "moran_i": None, "expected_i": None, "variance": None, "z_score": None,
                "p_value": None, "neighbor_summary": None, "status": "Not available", "message": str(exc),
            }
        push_residual_spatial_diagnostics(feedback, summary)

        out_fields = source.fields()
        out_fields.append(QgsField("model_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("model_used", QVariant.Int))

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
                out_feature.setAttribute("model_resid", float(residuals[result_map[fid]]))
                out_feature.setAttribute("model_used", 1)
            else:
                out_feature.setAttribute("model_resid", None)
                out_feature.setAttribute("model_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 70 * (current / total)))

        self._write_html(html_path, target_field, model_key, summary, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats model residual spatial autocorrelation output",
            {
                "model_resid": "Observed minus predicted (in-sample) from the fitted model",
                "model_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            "model_residual_autocorrelation",
        )
        apply_renderer(layer, diverging_residual_renderer(layer, layer.geometryType(), "model_resid"))
        return {}

    def _write_html(self, path, target_field, model_key, summary, skipped):
        guidance = analyst_guidance_html(
            "Model Residual Spatial Autocorrelation Check",
            "Global Moran's I applied to a fitted model's in-sample residuals; "
            "detects leftover spatial structure the explanatory fields did not capture.",
            [
                "p-value is well above 0.05 (residuals are not distinguishable from spatial randomness).",
                "Moran's I is close to the expected value under randomness.",
            ],
            [
                "p-value below 0.05 with a positive Moran's I (residuals cluster - real spatial signal is being missed).",
                "A high model R2 combined with significant residual autocorrelation (fit looks good but is masking a spatial gap).",
            ],
            [
                "Add a distance-to-center or spatial-lag field to the explanatory set and refit.",
                "GWR / MGWR / Spatial Lag / Spatial Error Regression - models built to account for spatial structure directly.",
                "SHAP Spatial Attribution Map on the same model - see whether any field's contribution has an odd geographic pattern.",
            ],
            "Do not trust a high-R2 machine-learning model on spatial data "
            "until this check comes back clean - in-sample fit alone cannot "
            "tell you whether location itself is missing from the model.",
        )
        residual_values = summary.get("residual_values") or []
        residual_lag = summary.get("residual_lag") or []
        if residual_values:
            residual_chart = scatter_plot_svg(
                residual_values,
                residual_lag,
                x_label="Residual (observed - predicted)",
                y_label="Spatial lag of residual",
                trend_line=True,
                quadrant_shading=True,
            )
        else:
            residual_chart = "<p>Residual scatterplot unavailable for this run.</p>"
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Model Residual Spatial Autocorrelation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 8px 10px; text-align: left; font-size: .9rem; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
{chart_css()}
</style></head>
<body><div class="container">
<h1>Model Residual Spatial Autocorrelation Check</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Model: <strong>{html.escape(CV_MODEL_LABELS[model_key])}</strong></p>
<div class="summary">Skipped {skipped} record(s) with missing or non-numeric values.</div>
{residual_spatial_autocorrelation_html(summary)}
<h2>Residual Moran Scatterplot</h2>
{residual_chart}
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
