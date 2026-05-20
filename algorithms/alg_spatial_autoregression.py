# -*- coding: utf-8 -*-
"""Spatial Autoregression Processing Algorithm."""
from __future__ import annotations

import html
import logging
import math
import os
import tempfile

import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsGraduatedSymbolRenderer,
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
    QgsProject,
    QgsRendererRange,
    QgsSymbol,
)

from ..core.analysis_diagnostics import (
    crs_unit_warning,
    diagnostics_html,
    filter_weights_to_valid_ids,
    format_number,
    neighbor_summary,
    numeric_quality_summary,
    push_diagnostics,
    regression_quality_html,
    regression_quality_summary,
)
from ..core.weights import build_weights_matrix

logger = logging.getLogger("PlanX GeoStats Lab")


class SpatialAutoregressionAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    DEP_VAR = "DEP_VAR"
    INDEPENDENTS = "INDEPENDENTS"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    WEIGHT_LABELS = ["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"]
    WEIGHT_KEYS = ["queen", "rook", "knn", "distance"]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "spatial_autoregression"

    def displayName(self) -> str:
        return "Spatial Autoregression (Spatial Lag Model)"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def createInstance(self):
        return SpatialAutoregressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a global spatial lag regression model using PySAL spreg. The model "
            "estimates a dependent variable as a function of explanatory variables and "
            "the spatially lagged dependent variable, y = rho W y + X beta + error.\n\n"
            "Use this tool when OLS residuals show spatial autocorrelation and the "
            "planning question suggests a neighborhood spillover process. The output "
            "layer includes fitted values, residuals, standardized residuals, spatial "
            "lag values, neighbor counts, and a complete-record audit flag. The HTML "
            "report summarizes coefficient estimates, spatial rho, model fit, weight "
            "support, and analyst caveats.\n\n"
            "This method requires libpysal and spreg in the active QGIS Python "
            "environment. Open PlanX GeoStats Lab > GeoStats Libraries if the "
            "dependencies are missing."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.TypeVectorAnyGeometry],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.DEP_VAR,
                "Dependent variable field",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.INDEPENDENTS,
                "Explanatory variable fields",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial relationship / weights type",
                options=self.WEIGHT_LABELS,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of neighbors (K value, KNN only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=8,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                "Distance band threshold (map units, Distance Band only)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                minValue=0.0001,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output spatial lag model layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output spatial autoregression HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Spatial autoregression diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        libpysal, ml_lag = self._load_spreg_dependencies()
        dep_var = self.parameterAsString(parameters, self.DEP_VAR, context)
        indep_fields = self.parameterAsFields(parameters, self.INDEPENDENTS, context)
        if not indep_fields:
            raise QgsProcessingException("At least one explanatory variable must be selected.")

        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_type = self.WEIGHT_KEYS[weight_type_idx]
        weight_label = self.WEIGHT_LABELS[weight_type_idx]
        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_spatial_autoregression_report.html")

        fields = source.fields()
        dep_idx = fields.lookupField(dep_var)
        exp_indices = [fields.lookupField(name) for name in indep_fields]
        if dep_idx < 0:
            raise QgsProcessingException(f"Dependent field '{dep_var}' not found.")
        missing = [name for name, idx in zip(indep_fields, exp_indices) if idx < 0]
        if missing:
            raise QgsProcessingException(f"Explanatory fields not found: {', '.join(missing)}")

        feedback.pushInfo("Extracting complete numeric records for spatial autoregression...")
        y_values = []
        x_rows = []
        valid_fids = []
        valid_dep_map = {}
        total = source.featureCount() or 1
        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            dep_value = self._to_float(feature.attribute(dep_idx))
            exp_values = [self._to_float(feature.attribute(field_idx)) for field_idx in exp_indices]
            if dep_value is None or any(value is None for value in exp_values):
                continue
            y_values.append(dep_value)
            x_rows.append(exp_values)
            valid_fids.append(int(feature.id()))
            valid_dep_map[int(feature.id())] = dep_value
            feedback.setProgress(int(20 * (idx / total)))

        n = len(y_values)
        p = len(indep_fields)
        if n <= p + 2:
            raise QgsProcessingException(
                f"Insufficient complete records ({n}). Spatial lag regression needs more observations "
                f"than explanatory variables, intercept, and spatial rho ({p + 2} parameters)."
            )

        y = np.array(y_values, dtype=float)
        x_data = np.array(x_rows, dtype=float)
        if float(np.std(y)) <= 1e-12:
            raise QgsProcessingException("The dependent variable is constant across complete records.")

        feedback.pushInfo(f"Building {weight_label} spatial weights...")
        neighbors, _, _, _ = build_weights_matrix(
            source,
            weight_type,
            k_neighbors=k_neighbors,
            distance_band=distance_band,
            feedback=feedback,
        )
        filtered_neighbors, filtered_weights, valid_id_order = filter_weights_to_valid_ids(neighbors, valid_fids)
        n_summary = neighbor_summary(filtered_neighbors, valid_id_order)
        numeric_summary = numeric_quality_summary(source.featureCount(), valid_dep_map, y)
        crs_warning = crs_unit_warning(source)
        push_diagnostics(feedback, numeric_summary, n_summary, crs_warning)

        total_links = sum(len(row) for row in filtered_neighbors.values())
        if total_links == 0:
            raise QgsProcessingException(
                "No valid spatial neighbor links remain after filtering complete records. "
                "Increase the distance band, increase K, or review contiguity geometry before fitting a spatial model."
            )
        if n_summary["isolated"] > 0:
            feedback.pushWarning(
                "The spatial lag model includes isolated observations with no valid neighbors. "
                "Review the weight choice before using coefficients for planning decisions."
            )

        quality = regression_quality_summary(y, x_data, indep_fields, source.featureCount())
        for risk in quality["risks"]:
            feedback.pushWarning(risk)

        try:
            w = libpysal.weights.W(filtered_neighbors, filtered_weights, id_order=valid_id_order)
            w.transform = "r"
        except Exception as exc:
            raise QgsProcessingException(f"Could not build a libpysal W object for spatial regression: {exc}")

        try:
            islands = list(getattr(w, "islands", []))
            if islands:
                feedback.pushWarning(f"{len(islands)} observation(s) are islands in the libpysal weights object.")
        except Exception:
            islands = []

        feedback.pushInfo("Fitting spatial lag regression with spreg.ML_Lag...")
        try:
            model = ml_lag(
                y.reshape((-1, 1)),
                x_data,
                w=w,
                name_y=dep_var,
                name_x=list(indep_fields),
                name_w=weight_label,
            )
        except Exception as exc:
            raise QgsProcessingException(
                "Spatial autoregression failed during model estimation. "
                "Review complete-record count, multicollinearity, weight density, and isolated observations. "
                f"Underlying error: {exc}"
            )

        results = self._extract_model_results(model, y, x_data, indep_fields)
        lag_y = self._spatial_lag(y, filtered_neighbors, filtered_weights, valid_id_order)
        residual_std = float(np.std(results["residuals"]))
        if residual_std > 0:
            std_residuals = results["residuals"] / residual_std
        else:
            std_residuals = np.zeros(n)

        feedback.pushInfo(
            "Spatial lag model fitted: "
            f"rho={format_number(results['rho'], 6)}, "
            f"pseudo R2={format_number(results['pseudo_r2'], 6)}, "
            f"AIC={format_number(results['aic'], 4)}."
        )

        out_fields = source.fields()
        out_fields.append(QgsField("sar_pred", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("sar_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("sar_stdres", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("sar_lag_y", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("sar_nbrs", QVariant.Int))
        out_fields.append(QgsField("sar_used", QVariant.Int))

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs(),
        )
        self.out_layer_id = dest_id

        valid_set = set(valid_id_order)
        result_map = {}
        for row_idx, fid in enumerate(valid_id_order):
            result_map[fid] = {
                "pred": results["predicted"][row_idx],
                "resid": results["residuals"][row_idx],
                "stdres": std_residuals[row_idx],
                "lag_y": lag_y[row_idx],
                "neighbors": len([nid for nid in filtered_neighbors.get(fid, []) if nid in valid_set]),
            }

        feedback.pushInfo("Writing spatial autoregression audit fields...")
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = int(feature.id())
            row = result_map.get(fid)
            if row is None:
                out_feature.setAttribute("sar_pred", None)
                out_feature.setAttribute("sar_resid", None)
                out_feature.setAttribute("sar_stdres", None)
                out_feature.setAttribute("sar_lag_y", None)
                out_feature.setAttribute("sar_nbrs", None)
                out_feature.setAttribute("sar_used", 0)
            else:
                out_feature.setAttribute("sar_pred", float(row["pred"]))
                out_feature.setAttribute("sar_resid", float(row["resid"]))
                out_feature.setAttribute("sar_stdres", float(row["stdres"]))
                out_feature.setAttribute("sar_lag_y", float(row["lag_y"]))
                out_feature.setAttribute("sar_nbrs", int(row["neighbors"]))
                out_feature.setAttribute("sar_used", 1)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(55 + 45 * (current / total)))

        self._write_html(
            html_path,
            dep_var,
            indep_fields,
            weight_label,
            weight_type,
            k_neighbors,
            distance_band,
            results,
            quality,
            numeric_summary,
            n_summary,
            crs_warning,
            islands,
        )
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _load_spreg_dependencies(self):
        try:
            import libpysal
            from spreg import ML_Lag
        except Exception as exc:
            raise QgsProcessingException(
                "Spatial Autoregression requires optional libraries libpysal and spreg. "
                "Open PlanX GeoStats Lab > GeoStats Libraries to review the active QGIS Python "
                "environment and install or update GeoStats libraries with explicit approval. "
                f"Import error: {exc}"
            )
        return libpysal, ML_Lag

    def _to_float(self, value):
        if value is None or value == QVariant() or str(value) == "NULL":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _extract_model_results(self, model, y, x_data, indep_fields):
        n = len(y)
        betas = np.asarray(getattr(model, "betas", []), dtype=float).flatten()
        if betas.size == 0:
            raise QgsProcessingException("spreg returned no coefficient estimates.")

        predicted = np.asarray(getattr(model, "predy", np.full(n, np.nan)), dtype=float).flatten()
        if predicted.size != n or not np.all(np.isfinite(predicted)):
            predicted = np.full(n, np.nan)
        residuals = np.asarray(getattr(model, "u", y - predicted), dtype=float).flatten()
        if residuals.size != n or not np.all(np.isfinite(residuals)):
            residuals = y - predicted

        std_errors = np.asarray(getattr(model, "std_err", np.full(betas.size, np.nan)), dtype=float).flatten()
        if std_errors.size != betas.size:
            std_errors = np.full(betas.size, np.nan)

        z_values, p_values = self._extract_z_stats(getattr(model, "z_stat", []), betas.size)
        names = ["Intercept"] + list(indep_fields)
        if len(names) < betas.size:
            names.append("Spatial lag rho")
        while len(names) < betas.size:
            names.append(f"Parameter {len(names) + 1}")
        names = names[:betas.size]

        return {
            "coefficients": betas,
            "std_errors": std_errors,
            "z_values": z_values,
            "p_values": p_values,
            "names": names,
            "predicted": predicted,
            "residuals": residuals,
            "rho": self._safe_float(getattr(model, "rho", betas[-1] if betas.size else None)),
            "pseudo_r2": self._safe_float(getattr(model, "pr2", None)),
            "log_likelihood": self._safe_float(getattr(model, "logll", None)),
            "aic": self._safe_float(getattr(model, "aic", None)),
            "schwarz": self._safe_float(getattr(model, "schwarz", None)),
            "n": n,
            "p": int(x_data.shape[1]),
        }

    def _extract_z_stats(self, z_stat, size: int):
        z_values = np.full(size, np.nan)
        p_values = np.full(size, np.nan)
        for idx, item in enumerate(list(z_stat)[:size]):
            try:
                z_values[idx] = float(item[0])
                p_values[idx] = float(item[1])
            except Exception:
                continue
        return z_values, p_values

    def _safe_float(self, value):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return result

    def _spatial_lag(self, y, neighbors, weights, id_order):
        id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
        lag = np.zeros(len(id_order), dtype=float)
        for idx, fid in enumerate(id_order):
            total = 0.0
            for j, nid in enumerate(neighbors.get(fid, [])):
                if nid in id_to_idx:
                    row_weights = weights.get(fid, [])
                    weight = row_weights[j] if j < len(row_weights) else 0.0
                    total += weight * y[id_to_idx[nid]]
            lag[idx] = total
        return lag

    def _write_html(
        self,
        path,
        dep_var,
        indep_fields,
        weight_label,
        weight_type,
        k_neighbors,
        distance_band,
        results,
        quality,
        numeric_summary,
        n_summary,
        crs_warning,
        islands,
    ):
        coefficient_rows = []
        for name, coef, se, z_value, p_value in zip(
            results["names"],
            results["coefficients"],
            results["std_errors"],
            results["z_values"],
            results["p_values"],
        ):
            coefficient_rows.append(
                "<tr>"
                f"<td><strong>{html.escape(name)}</strong></td>"
                f"<td>{format_number(coef, 6)}</td>"
                f"<td>{format_number(se, 6)}</td>"
                f"<td>{format_number(z_value, 4)}</td>"
                f"<td>{format_number(p_value, 6)}</td>"
                "</tr>"
            )

        rho_text = "positive"
        rho = results["rho"]
        if rho is None:
            rho_text = "not reported"
        elif rho < 0:
            rho_text = "negative"
        elif abs(rho) < 1e-9:
            rho_text = "near zero"

        weight_detail = weight_label
        if weight_type == "knn":
            weight_detail = f"{weight_label}, K={k_neighbors}"
        elif weight_type == "distance":
            weight_detail = f"{weight_label}, threshold={distance_band} map units"

        island_note = ""
        if islands:
            island_note = (
                "<div class=\"note\"><strong>Weight warning:</strong> "
                f"{len(islands)} observation(s) are islands with no valid neighbors. "
                "Review the spatial relationship before treating rho as a stable planning signal.</div>"
            )

        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Spatial Autoregression</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #25313f; background: #f6f8fb; margin: 0; padding: 24px; line-height: 1.55; }}
.container {{ max-width: 1120px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.72rem; color: #17212f; }}
h2 {{ color: #1a202c; font-size: 1.15rem; margin: 28px 0 12px; border-left: 4px solid #2b6cb0; padding-left: 10px; }}
.subtitle {{ color: #607086; margin: 0 0 24px; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 16px 18px; margin: 20px 0; }}
.note {{ background: #fff8e6; border-left: 5px solid #b7791f; padding: 14px 18px; margin: 20px 0; }}
.next-action {{ background: #f0fff4; border-left: 5px solid #2f855a; padding: 16px 18px; border-radius: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 18px 0 24px; }}
.card {{ border: 1px solid #d9e2ec; background: #f8fafc; border-radius: 6px; padding: 14px; }}
.card-title {{ color: #607086; font-size: .74rem; text-transform: uppercase; font-weight: 700; }}
.card-value {{ color: #1f5f8b; font-size: 1.28rem; font-weight: 800; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px; text-align: left; vertical-align: top; font-size: .87rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
.metric-name {{ font-weight: 700; color: #314155; }}
footer {{ margin-top: 36px; padding-top: 14px; border-top: 1px solid #edf2f7; color: #7a899c; font-size: .82rem; }}
</style>
</head>
<body>
<div class="container">
<h1>Spatial Autoregression (Spatial Lag Model)</h1>
<p class="subtitle">Dependent field: <strong>{html.escape(dep_var)}</strong> | Explanatory variables: <strong>{html.escape(', '.join(indep_fields))}</strong> | Weights: <strong>{html.escape(weight_detail)}</strong></p>

<section class="summary">
<strong>Executive summary.</strong> This spatial lag model estimates whether nearby values of the dependent variable help explain the target outcome after controlling for the selected explanatory variables. A {html.escape(rho_text)} rho indicates the direction of neighborhood spillover in the fitted model. Use the coefficient table together with the neighbor diagnostics below before translating this result into policy, zoning, service-area, or investment decisions.
</section>

<div class="grid">
<div class="card"><div class="card-title">Complete Records</div><div class="card-value">{results['n']}</div></div>
<div class="card"><div class="card-title">Spatial Rho</div><div class="card-value">{format_number(results['rho'], 6)}</div></div>
<div class="card"><div class="card-title">Pseudo R2</div><div class="card-value">{format_number(results['pseudo_r2'], 6)}</div></div>
<div class="card"><div class="card-title">AIC</div><div class="card-value">{format_number(results['aic'], 4)}</div></div>
</div>

<h2>Method Assumptions</h2>
<p>The fitted model follows <strong>y = rho W y + X beta + error</strong>. The matrix W is the selected row-standardized spatial relationship, rho estimates the strength of the spatially lagged dependent variable, and beta estimates the global relationship between each explanatory variable and the dependent variable after accounting for that spatial lag.</p>
<p>This is a global model. It does not replace local diagnostics, residual review, or domain judgment about boundaries, network barriers, planning zones, market areas, or environmental discontinuities.</p>

{diagnostics_html(numeric_summary, n_summary, crs_warning)}
{regression_quality_html(quality)}
{island_note}

<h2>Coefficient Estimates</h2>
<table>
<thead><tr><th>Parameter</th><th>Coefficient</th><th>Std Error</th><th>z-statistic</th><th>p-value</th></tr></thead>
<tbody>{''.join(coefficient_rows)}</tbody>
</table>

<h2>Model Fit Statistics</h2>
<table>
<tbody>
<tr><td class="metric-name">Log likelihood</td><td>{format_number(results['log_likelihood'], 6)}</td></tr>
<tr><td class="metric-name">Akaike Information Criterion</td><td>{format_number(results['aic'], 6)}</td></tr>
<tr><td class="metric-name">Schwarz Criterion</td><td>{format_number(results['schwarz'], 6)}</td></tr>
<tr><td class="metric-name">Minimum neighbors</td><td>{n_summary['minimum']}</td></tr>
<tr><td class="metric-name">Median neighbors</td><td>{n_summary['median']:.2f}</td></tr>
<tr><td class="metric-name">Maximum neighbors</td><td>{n_summary['maximum']}</td></tr>
</tbody>
</table>

<h2>Interpretation</h2>
<p>A positive and substantively meaningful rho suggests that high or low nearby dependent-variable values reinforce the modeled outcome after the explanatory variables are considered. In planning terms, that can point to spillover, contagion, shared service access, market clustering, or omitted spatial process. A negative rho suggests local contrast, edge effects, displacement, or competing spatial relationships. A near-zero or statistically weak rho means the selected weight graph may not add useful spatial dependence beyond the explanatory variables.</p>

<h2>Recommended Analyst Action</h2>
<div class="next-action">Compare this model with OLS and GLR outputs, inspect standardized residuals on the map, and rerun the model with an alternative defensible weight definition if the neighbor graph is sparse, very dense, or strongly sensitive to K or distance-band settings. Treat rho as a planning signal only when the weight definition matches the real spatial process being studied.</div>

<h2>Caveats</h2>
<ul>
<li>Spatial autoregression is sensitive to how W is defined. Contiguity, KNN, and distance-band weights answer different planning questions.</li>
<li>Distance-band models should use a projected CRS with meaningful map units. Geographic degrees can produce misleading thresholds.</li>
<li>High multicollinearity, small sample size, islands, and unstable neighborhood support can make coefficients difficult to interpret.</li>
<li>The model is global. Spatially varying relationships may still require GWR or future MGWR-style analysis.</li>
</ul>

<footer>Generated by PlanX GeoStats Lab spatial statistics engine.</footer>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}
        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}

        feedback.pushInfo("Applying spatial autoregression standardized residual styling...")
        ranges = []
        range_definitions = [
            (-9999.0, -2.5, "#2166ac", "< -2.5 Std Residual"),
            (-2.5, -1.5, "#67a9cf", "-2.5 to -1.5 Std Residual"),
            (-1.5, -0.5, "#d1e5f0", "-1.5 to -0.5 Std Residual"),
            (-0.5, 0.5, "#f7f7f7", "-0.5 to 0.5 Std Residual"),
            (0.5, 1.5, "#fddbc7", "0.5 to 1.5 Std Residual"),
            (1.5, 2.5, "#f4a582", "1.5 to 2.5 Std Residual"),
            (2.5, 9999.0, "#b2182b", "> 2.5 Std Residual"),
        ]
        for min_value, max_value, color_hex, label in range_definitions:
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(QColor(color_hex))
            symbol.setOpacity(0.85)
            if symbol.symbolLayerCount() > 0:
                symbol_layer = symbol.symbolLayer(0)
                if hasattr(symbol_layer, "setStrokeColor"):
                    symbol_layer.setStrokeColor(QColor("#b0b0b0"))
                if hasattr(symbol_layer, "setStrokeWidth"):
                    symbol_layer.setStrokeWidth(0.1)
                if hasattr(symbol_layer, "setOutlineColor"):
                    symbol_layer.setOutlineColor(QColor("#b0b0b0"))
            ranges.append(QgsRendererRange(min_value, max_value, symbol, label))
        layer.setRenderer(QgsGraduatedSymbolRenderer("sar_stdres", ranges))
        layer.triggerRepaint()
        return {}
