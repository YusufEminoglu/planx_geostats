# -*- coding: utf-8 -*-
"""TabPFN Regression (Tabular Foundation Model) Processing Algorithm."""
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
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.ml_engines import extract_regression_matrix, fit_tabpfn_regressor
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, scatter_plot_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class TabPFNRegressionAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    MAX_ROWS = 10000
    MAX_FEATURES = 500

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "tabpfn_regression"

    def displayName(self) -> str:
        return "TabPFN Regression (Tabular Foundation Model)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("tabpfn_regression")

    def createInstance(self):
        return TabPFNRegressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a zero-shot regression prediction with TabPFN v2 (Hollmann "
            "et al., 'Accurate predictions on small data with a tabular "
            "foundation model', Nature, 2025) - a transformer pretrained once, "
            "offline, on millions of synthetically generated tabular learning "
            "problems. At call time it performs in-context learning: your "
            "entire training table is passed through the network as a single "
            "forward pass together with the records to predict, with no "
            "gradient-descent training loop of its own the way every other "
            "model in this group has (Random Forest grows trees, MLP updates "
            "weights over many epochs; TabPFN does neither - it has already "
            "'learned how to learn' from its pretraining corpus and simply "
            "reads your table as a prompt).\n\n"
            "This is the newest and most experimental engine in the group: on "
            "the small-to-medium tabular benchmarks it was evaluated on, TabPFN "
            "v2 matched or beat tuned Gradient Boosting while fitting in "
            "seconds with no hyperparameter search - but it has an explicit, "
            "hard operating envelope of up to 10,000 rows and 500 features "
            "(enforced here; larger selections raise a clear error rather than "
            "a silent quality drop), needs an internet connection the first "
            "time it runs on this machine to download its pretrained weights "
            "(cached afterward), and its behavior on planning-domain data "
            "specifically has far less accumulated field experience behind it "
            "than Random Forest or Gradient Boosting.\n\n"
            "Output: fitted values and residuals per complete record, plus an "
            "HTML report with R2, RMSE, and MAE. TabPFN provides no native "
            "feature importances - pair it with Permutation Feature Importance "
            "or SHAP Global Feature Importance when the explanatory question "
            "matters as much as raw accuracy, and always cross-check its "
            "out-of-sample behavior with the Spatial k-Fold Cross-Validation "
            "Evaluator before trusting it for a planning decision - a model "
            "with no tunable hyperparameters is not a model that can be "
            "assumed correct.\n\n"
            "Requires the optional tabpfn package (Setup and Diagnostics > "
            "Install / Update GeoStats Libraries) and, on first use, an "
            "internet connection.\n\n"
            "First use also needs a one-time TabPFN license acceptance. On "
            "Windows, TabPFN's interactive browser-login prompt cannot "
            "complete inside QGIS's embedded console, so accept the license "
            "once outside QGIS instead: open https://ux.priorlabs.ai/account, "
            "log in or register, accept the license, copy the API key, then "
            "set it as a permanent Windows environment variable named "
            "TABPFN_TOKEN and restart QGIS. After that, TabPFN runs with no "
            "further prompts."
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
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output TabPFN predictions layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output TabPFN HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "TabPFN regression report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")
        if len(feature_fields) > self.MAX_FEATURES:
            raise QgsProcessingException(
                f"TabPFN supports up to {self.MAX_FEATURES} features; {len(feature_fields)} were selected."
            )

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_tabpfn_regression_report.html")

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
        if len(y) > self.MAX_ROWS:
            raise QgsProcessingException(
                f"TabPFN supports up to {self.MAX_ROWS:,} records; this layer has {len(y):,} complete "
                "records. Use Random Forest Regression, a Gradient Boosting engine, or another "
                "tool in this group for larger tables."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric values.")

        feedback.pushInfo("Running TabPFN in-context inference (downloading pretrained weights on first use)...")
        try:
            results = fit_tabpfn_regressor(x, y)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("TabPFN Regression", ["tabpfn"], exc))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        feedback.pushInfo(f"R2={results['r2']:.4f}, RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}")

        out_fields = source.fields()
        out_fields.append(QgsField("tabpfn_pred", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("tabpfn_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("tabpfn_used", QVariant.Int))

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
                out_feature.setAttribute("tabpfn_pred", float(results["fitted"][row_idx]))
                out_feature.setAttribute("tabpfn_resid", float(results["residuals"][row_idx]))
                out_feature.setAttribute("tabpfn_used", 1)
            else:
                out_feature.setAttribute("tabpfn_pred", None)
                out_feature.setAttribute("tabpfn_resid", None)
                out_feature.setAttribute("tabpfn_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats TabPFN regression output",
            {
                "tabpfn_pred": "TabPFN in-context predicted value",
                "tabpfn_resid": "Observed minus predicted (positive = model underpredicted)",
                "tabpfn_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            "tabpfn_regression",
        )
        return {}

    def _write_html(self, path, target_field, feature_fields, results, skipped):
        residual_chart = scatter_plot_svg(
            list(results["fitted"]),
            list(results["residuals"]),
            x_label="Fitted value",
            y_label="Residual",
            trend_line=True,
            split_y=0.0,
        )
        guidance = analyst_guidance_html(
            "TabPFN Regression",
            "A pretrained tabular foundation model performing in-context "
            "learning (no training loop, no hyperparameters to tune) - the "
            "newest and least field-tested engine in this group.",
            [
                "The record count and feature count both sit comfortably inside TabPFN's published operating envelope.",
                "R2 is cross-checked against Random Forest Regression / Gradient Boosting Regression on the same fields, not taken alone.",
                "Out-of-sample performance was verified with the Spatial k-Fold Cross-Validation Evaluator before use in a planning decision.",
            ],
            [
                "R2 is markedly better than every tree-based engine with no clear explanation (worth a second look for target leakage before celebrating).",
                "The dataset sits near the 10,000-row or 500-feature ceiling (results near a hard operating boundary deserve extra scrutiny).",
                "Predictions are used for a consequential decision without any out-of-sample check.",
            ],
            [
                "Gradient Boosting Regression (CatBoost / XGBoost / LightGBM / scikit-learn) - the established, thoroughly field-tested comparison point.",
                "Permutation Feature Importance / SHAP Global Feature Importance - TabPFN itself gives no feature ranking.",
                "Spatial k-Fold Cross-Validation Evaluator - honest out-of-sample R2, essential given how little planning-domain track record this engine has.",
            ],
            "Treat TabPFN as a fast, competitive second opinion alongside "
            "Gradient Boosting rather than a default first choice, until it "
            "has accumulated more field experience on planning-specific data.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TabPFN Regression Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
{chart_css()}
</style></head>
<body><div class="container">
<h1>TabPFN Regression (Tabular Foundation Model)</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Explanatory fields: <strong>{html.escape(', '.join(feature_fields))}</strong></p>
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
