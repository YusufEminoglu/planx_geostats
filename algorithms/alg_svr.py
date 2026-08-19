# -*- coding: utf-8 -*-
"""Support Vector Regression Processing Algorithm."""
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
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.symbology import apply_renderer, diverging_residual_renderer
from ..core.ml_engines import extract_regression_matrix, fit_svr
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, scatter_plot_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class SVRAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    KERNEL = "KERNEL"
    C = "C"
    EPSILON = "EPSILON"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    KERNELS = ["Linear", "RBF (Gaussian, default)", "Polynomial (degree 3)"]
    KERNEL_KEYS = ["linear", "rbf", "poly"]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "support_vector_regression"

    def displayName(self) -> str:
        return "Support Vector Regression (SVR)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("support_vector_regression")

    def createInstance(self):
        return SVRAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Support Vector Regression model: finds a function that stays "
            "within an epsilon-wide tolerance tube around as many training points "
            "as possible, using only the points outside that tube (the support "
            "vectors) to shape the fit. With the RBF kernel it can capture smooth "
            "non-linear relationships with few tunable parameters; with the Linear "
            "kernel it behaves similarly to a robust linear regression.\n\n"
            "Features and the target are standardized internally before fitting "
            "(SVR is scale-sensitive) and predictions are converted back to the "
            "original target scale automatically - no manual standardization is "
            "needed.\n\n"
            "Output: fitted values and residuals per complete record, plus an HTML "
            "report with R2, RMSE, and MAE. SVR does not produce feature "
            "importances or coefficients, so it answers 'how well can this be "
            "predicted' rather than 'which field matters most' - pair it with "
            "Random Forest Regression or SHAP-based tools when the explanatory "
            "question matters as much as predictive accuracy.\n\n"
            "Epsilon controls the width of the no-penalty tolerance tube: a larger "
            "epsilon ignores small residuals and produces a simpler, flatter fit; "
            "a smaller epsilon fits more closely but risks overfitting noise. C "
            "controls how strongly points outside the tube are penalized - raise "
            "it to fit training data more tightly, lower it for a smoother, more "
            "regularized fit."
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
        self.addParameter(QgsProcessingParameterEnum(self.KERNEL, "Kernel", options=self.KERNELS, defaultValue=1))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.C, "Regularization (C)", type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0, minValue=0.001, maxValue=1000.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.EPSILON, "Tolerance tube width (epsilon)", type=QgsProcessingParameterNumber.Double,
                defaultValue=0.1, minValue=0.0, maxValue=10.0,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output SVR predictions layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output SVR HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "SVR diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        kernel = self.KERNEL_KEYS[self.parameterAsEnum(parameters, self.KERNEL, context)]
        c_value = self.parameterAsDouble(parameters, self.C, context)
        epsilon = self.parameterAsDouble(parameters, self.EPSILON, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_svr_report.html")

        feedback.pushInfo("Extracting complete numeric records...")
        try:
            extraction = extract_regression_matrix(source, feature_fields, target_field, feedback, (0, 30))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, valid_fids, skipped = extraction["x"], extraction["y"], extraction["valid_fids"], extraction["skipped"]
        if len(y) <= len(feature_fields) + 1:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {len(feature_fields)} explanatory field(s)."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric values.")

        feedback.pushInfo(f"Fitting SVR ({kernel} kernel)...")
        try:
            results = fit_svr(x, y, kernel=kernel, c=c_value, epsilon=epsilon)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Support Vector Regression", ["scikit-learn"], exc))

        feedback.pushInfo(f"R2={results['r2']:.4f}, RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}")

        out_fields = source.fields()
        out_fields.append(QgsField("svr_pred", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("svr_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("svr_used", QVariant.Int))

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
                out_feature.setAttribute("svr_pred", float(results["fitted"][row_idx]))
                out_feature.setAttribute("svr_resid", float(results["residuals"][row_idx]))
                out_feature.setAttribute("svr_used", 1)
            else:
                out_feature.setAttribute("svr_pred", None)
                out_feature.setAttribute("svr_resid", None)
                out_feature.setAttribute("svr_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, kernel, c_value, epsilon, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats Support Vector Regression output",
            {
                "svr_pred": "SVR predicted value (converted back from the internally standardized fit)",
                "svr_resid": "Observed minus predicted (positive = model underpredicted)",
                "svr_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            "support_vector_regression",
        )
        apply_renderer(layer, diverging_residual_renderer(layer, layer.geometryType(), "svr_resid"))
        return {}

    def _write_html(self, path, target_field, feature_fields, kernel, c_value, epsilon, results, skipped):
        residual_chart = scatter_plot_svg(
            list(results["fitted"]),
            list(results["residuals"]),
            x_label="Fitted value",
            y_label="Residual",
            trend_line=True,
            split_y=0.0,
        )
        guidance = analyst_guidance_html(
            "Support Vector Regression",
            "A tolerance-tube fit shaped only by the points that fall outside "
            "it (the support vectors); strong for smooth non-linear relationships, "
            "gives no feature-level explanation on its own.",
            [
                "R2 is comparable to or better than a linear alternative (GLR).",
                "Kernel choice matches the expected relationship shape (RBF for smooth curvature, Linear for near-linear).",
            ],
            [
                "R2 much lower than Random Forest Regression on the same fields (relationship may be too irregular for this kernel).",
                "Predictions cluster near the mean (C too low or epsilon too wide).",
            ],
            [
                "Random Forest Regression - compare fit and get feature importances SVR cannot provide.",
                "SHAP Global Feature Importance - explain predictions from any of these models.",
                "Spatial k-Fold Cross-Validation Evaluator - honest out-of-sample R2.",
            ],
            "Use SVR when the predictive question (how accurately can this be "
            "estimated) matters more than the explanatory question (which field "
            "drives it); pair with a tree-based tool for the latter.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Support Vector Regression Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
{chart_css()}
</style></head>
<body><div class="container">
<h1>Support Vector Regression (SVR)</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Explanatory fields: <strong>{html.escape(', '.join(feature_fields))}</strong></p>
<p>Kernel: <strong>{html.escape(kernel)}</strong> | C: <strong>{c_value:g}</strong> | Epsilon: <strong>{epsilon:g}</strong></p>
<div class="summary">
R2 = {results['r2']:.6f} | RMSE = {results['rmse']:.6f} | MAE = {results['mae']:.6f}<br>
Skipped {skipped} record(s) with missing or non-numeric values.
</div>
<h2>Residual vs. Fitted</h2>
{residual_chart}
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
