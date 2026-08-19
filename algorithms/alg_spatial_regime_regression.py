# -*- coding: utf-8 -*-
"""Spatial Regime Regression Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
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
from ..core.symbology import apply_renderer, diverging_residual_renderer
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, scatter_plot_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class SpatialRegimeRegressionAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    DEP_VAR = "DEP_VAR"
    INDEPENDENTS = "INDEPENDENTS"
    REGIME_FIELD = "REGIME_FIELD"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "spatial_regime_regression"

    def displayName(self) -> str:
        return "Spatial Regime Regression"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def icon(self):
        return algorithm_icon("spatial_regime_regression")

    def createInstance(self):
        return SpatialRegimeRegressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits an OLS regression with a fully separate intercept and set "
            "of coefficients for every distinct value of Regime field (a "
            "categorical field defining sub-areas - districts, zoning "
            "categories, north/south halves of the study area, anything "
            "that plausibly changes how the explanatory fields relate to "
            "the outcome) via spreg's regimes framework, then runs a Chow "
            "test asking whether those separate-per-regime coefficients fit "
            "meaningfully better than one pooled model would.\n\n"
            "This answers a structural-instability question that a single "
            "OLS run with the regime field as a plain dummy variable cannot: "
            "not just 'does the regime shift the average outcome' but 'does "
            "the whole relationship between the explanatory fields and the "
            "outcome differ by regime'. Requires the optional libpysal and "
            "spreg packages (Setup and Diagnostics > Install / Update "
            "GeoStats Libraries).\n\n"
            "Output: fitted values and residuals per complete record, plus "
            "an HTML report with one coefficient table per regime and the "
            "joint Chow test (statistic, degrees of freedom, p-value) for "
            "overall structural instability across regimes.\n\n"
            "A significant Chow test (p < 0.05) means the regimes genuinely "
            "behave differently and separate models are justified; a "
            "non-significant one means a single pooled model (Generalized "
            "Linear Regression, without the regime split) is simpler and "
            "just as defensible. Each regime needs enough complete records "
            "on its own to estimate every coefficient - a rare regime "
            "category will produce unstable, wide-ranging coefficients even "
            "if the test result looks clean overall."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.DEP_VAR, "Dependent variable field", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.INDEPENDENTS, "Explanatory variable fields", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric, allowMultiple=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.REGIME_FIELD, "Regime field (categorical)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output spatial regime regression predictions layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output spatial regime regression HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Spatial regime regression report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        dep_var = self.parameterAsString(parameters, self.DEP_VAR, context)
        indep_fields = self.parameterAsFields(parameters, self.INDEPENDENTS, context)
        regime_field = self.parameterAsString(parameters, self.REGIME_FIELD, context)
        if not indep_fields:
            raise QgsProcessingException("At least one explanatory variable must be selected.")
        if regime_field in indep_fields or regime_field == dep_var:
            raise QgsProcessingException("Regime field must be different from the dependent and explanatory fields.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_spatial_regime_report.html")

        fields = source.fields()
        dep_idx = fields.lookupField(dep_var)
        exp_indices = [fields.lookupField(name) for name in indep_fields]
        regime_idx = fields.lookupField(regime_field)
        if dep_idx < 0:
            raise QgsProcessingException(f"Dependent field '{dep_var}' not found.")
        if regime_idx < 0:
            raise QgsProcessingException(f"Regime field '{regime_field}' not found.")
        missing = [name for name, idx in zip(indep_fields, exp_indices) if idx < 0]
        if missing:
            raise QgsProcessingException(f"Explanatory fields not found: {', '.join(missing)}")

        y_values, x_rows, regimes, valid_fids = [], [], [], []
        skipped = 0
        total = source.featureCount() or 1
        feedback.pushInfo("Extracting complete records...")
        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            dep_value = self._to_float(feature.attribute(dep_idx))
            exp_values = [self._to_float(feature.attribute(field_idx)) for field_idx in exp_indices]
            regime_raw = feature.attribute(regime_idx)
            regime_missing = regime_raw is None or regime_raw == NULL or str(regime_raw) == "NULL"
            if dep_value is None or regime_missing or any(value is None for value in exp_values):
                skipped += 1
                continue
            y_values.append(dep_value)
            x_rows.append(exp_values)
            regimes.append(str(regime_raw))
            valid_fids.append(feature.id())
            feedback.setProgress(int(20 * (idx / total)))

        regime_labels = sorted(set(regimes))
        if len(regime_labels) < 2:
            raise QgsProcessingException(f"At least 2 distinct regime values are required; found {len(regime_labels)}.")
        min_records_needed = (len(indep_fields) + 1) * len(regime_labels) + 1
        if len(y_values) <= min_records_needed:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y_values)}) to fit {len(regime_labels)} "
                f"regimes with {len(indep_fields)} explanatory field(s) each."
            )

        y = np.array(y_values, dtype=float).reshape(-1, 1)
        x_data = np.array(x_rows, dtype=float)
        regime_counts = {label: regimes.count(label) for label in regime_labels}
        small_regimes = [label for label, count in regime_counts.items() if count <= len(indep_fields) + 2]
        if small_regimes:
            feedback.pushWarning(
                "Regime(s) with very few complete records (coefficients will be unstable): "
                + ", ".join(f"{label} (n={regime_counts[label]})" for label in small_regimes)
            )

        try:
            from spreg import OLS_Regimes
        except Exception as exc:
            raise QgsProcessingException(optional_dependency_error("Spatial Regime Regression", ["libpysal", "spreg"], exc))

        feedback.pushInfo(f"Fitting OLS_Regimes across {len(regime_labels)} regime(s): {', '.join(regime_labels)}...")
        try:
            model = OLS_Regimes(
                y, x_data, regimes=regimes, constant_regi="many", cols2regi="all",
                name_y=dep_var, name_x=indep_fields, name_regimes=regime_field,
            )
        except Exception as exc:
            raise QgsProcessingException(f"spreg OLS_Regimes failed to fit: {exc}")

        betas = np.asarray(getattr(model, "betas", []), dtype=float).flatten()
        predicted = np.asarray(getattr(model, "predy", np.full(len(y_values), np.nan)), dtype=float).flatten()
        if predicted.size != len(y_values):
            predicted = np.full(len(y_values), np.nan)
        residuals = np.asarray(getattr(model, "u", y.flatten() - predicted), dtype=float).flatten()
        r2 = getattr(model, "r2", None)
        chow_joint = getattr(getattr(model, "chow", None), "joint", None)

        chow_stat, chow_pvalue = (None, None)
        if chow_joint is not None:
            try:
                chow_stat, chow_pvalue = float(chow_joint[0]), float(chow_joint[1])
            except (TypeError, IndexError, ValueError):
                pass

        coefficient_names = list(getattr(model, "name_x_r", [])) or (["Intercept"] + list(indep_fields)) * len(regime_labels)
        feedback.pushInfo(
            f"R2={r2:.4f}" if isinstance(r2, (int, float)) else "R2 not available"
        )
        if chow_pvalue is not None:
            feedback.pushInfo(f"Joint Chow test: statistic={chow_stat:.4f}, p={chow_pvalue:.6f}")
            if chow_pvalue < 0.05:
                feedback.pushInfo("Regimes are significantly different (p < 0.05) - the split is statistically justified.")
            else:
                feedback.pushWarning("Regimes are NOT significantly different (p >= 0.05) - a pooled model may be preferable.")
        else:
            feedback.pushWarning("Chow test statistic was not available from this spreg version.")

        out_fields = source.fields()
        out_fields.append(QgsField("regi_fit", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("regi_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("regi_used", QVariant.Int))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        result_map = {fid: idx for idx, fid in enumerate(valid_fids)}
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            if fid in result_map:
                row_idx = result_map[fid]
                fitted_val = float(predicted[row_idx]) if row_idx < len(predicted) and np.isfinite(predicted[row_idx]) else None
                resid_val = float(residuals[row_idx]) if row_idx < len(residuals) and np.isfinite(residuals[row_idx]) else None
                out_feature.setAttribute("regi_fit", fitted_val)
                out_feature.setAttribute("regi_resid", resid_val)
                out_feature.setAttribute("regi_used", 1)
            else:
                out_feature.setAttribute("regi_fit", None)
                out_feature.setAttribute("regi_resid", None)
                out_feature.setAttribute("regi_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 70 * (current / total)))

        self._write_html(html_path, dep_var, indep_fields, regime_field, regime_labels, regime_counts, betas, coefficient_names, r2, chow_stat, chow_pvalue, skipped, predicted, residuals)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats spatial regime regression output",
            {
                "regi_fit": "Fitted value from this record's own regime-specific model",
                "regi_resid": "Observed minus predicted",
                "regi_used": "1 if the record had complete values and was used to fit and predict, 0 otherwise",
            },
            "spatial_regime_regression",
        )
        apply_renderer(layer, diverging_residual_renderer(layer, layer.geometryType(), "regi_resid"))
        return {}

    def _to_float(self, value):
        if value is None or value == NULL or str(value) == "NULL":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _write_html(self, path, dep_var, indep_fields, regime_field, regime_labels, regime_counts, betas, coefficient_names, r2, chow_stat, chow_pvalue, skipped, predicted, residuals):
        finite_mask = np.isfinite(predicted) & np.isfinite(residuals)
        residual_chart = scatter_plot_svg(
            predicted[finite_mask].tolist(),
            residuals[finite_mask].tolist(),
            x_label="Fitted value",
            y_label="Residual",
            trend_line=True,
            split_y=0.0,
        )
        coef_rows = "".join(
            f"<tr><td>{html.escape(str(name))}</td><td>{value:.6f}</td></tr>"
            for name, value in zip(coefficient_names, betas)
        )
        counts_text = ", ".join(f"{html.escape(label)} (n={count})" for label, count in regime_counts.items())
        r2_text = f"{r2:.6f}" if isinstance(r2, (int, float)) else "n/a"
        chow_text = (
            f"statistic={chow_stat:.4f}, p={chow_pvalue:.6f} "
            + ("(regimes significantly different)" if chow_pvalue < 0.05 else "(regimes not significantly different)")
            if chow_pvalue is not None else "not available from this spreg version"
        )
        guidance = analyst_guidance_html(
            "Spatial Regime Regression",
            "Fully separate intercept and coefficients per regime, tested "
            "against a pooled model via a joint Chow test.",
            [
                "Joint Chow test is significant (p < 0.05), justifying the regime split.",
                "Every regime has enough complete records to estimate its own coefficients reliably.",
            ],
            [
                "Chow test is not significant (a pooled GLR model is simpler and equally valid).",
                "One or more regimes have very few complete records (unstable coefficients despite a significant overall test).",
            ],
            [
                "Generalized Linear Regression - the pooled-model comparison point.",
                "Model Comparison Matrix - compare this against other regression tools formally.",
            ],
            "Report the Chow test result alongside the coefficients - "
            "per-regime coefficients without the significance test invite "
            "over-interpreting differences that could be sampling noise.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Spatial Regime Regression Report</title>
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
<h1>Spatial Regime Regression</h1>
<p>Dependent: <strong>{html.escape(dep_var)}</strong> | Explanatory fields: <strong>{html.escape(', '.join(indep_fields))}</strong> | Regime field: <strong>{html.escape(regime_field)}</strong></p>
<div class="summary">
Regimes: {counts_text}<br>
R2 = {r2_text} | Joint Chow test: {chow_text}<br>
Skipped {skipped} record(s) with missing or non-numeric values.
</div>
<h2>Coefficients (per regime, in spreg's regime-column order)</h2>
<table><thead><tr><th>Coefficient</th><th>Value</th></tr></thead><tbody>{coef_rows}</tbody></table>
<h2>Residual vs. Fitted</h2>
{residual_chart}
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
