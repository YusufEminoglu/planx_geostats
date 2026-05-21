# -*- coding: utf-8 -*-
"""Spatial Error Regression Processing Algorithm."""
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
    push_residual_spatial_diagnostics,
    regression_quality_html,
    regression_quality_summary,
    residual_spatial_autocorrelation_html,
    residual_spatial_autocorrelation_summary,
)
from ..core.weights import build_weights_matrix

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class SpatialErrorRegressionAlgorithm(QgsProcessingAlgorithm):
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
        return "spatial_error_regression"

    def displayName(self) -> str:
        return "Spatial Error Regression (SEM)"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def icon(self):
        return algorithm_icon("spatial_error_regression")

    def createInstance(self):
        return SpatialErrorRegressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a global Spatial Error Model using PySAL spreg. SEM models spatial "
            "dependence in the error term rather than adding a spatially lagged dependent "
            "variable. Use it when OLS or GLR residuals remain spatially autocorrelated "
            "and the analyst suspects omitted spatial processes, shared boundaries, "
            "unobserved neighborhood context, or spatially structured measurement error.\n\n"
            "The output layer includes fitted values, residuals, standardized residuals, "
            "neighbor counts, and complete-record audit flags. The HTML report summarizes "
            "lambda, coefficient estimates, model fit, input diagnostics, residual Moran's I, "
            "caveats, and recommended next actions.\n\n"
            "This method requires libpysal and spreg in the active QGIS Python environment. "
            "Run PlanX GeoStats Lab > 00 | Setup and Diagnostics > GeoStats Library Status or Install / Update GeoStats Libraries if the dependencies are missing."
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
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output spatial error model layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output spatial error regression HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Spatial error regression diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        libpysal, ml_error = self._load_spreg_dependencies()
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
            html_path = os.path.join(tempfile.gettempdir(), "planx_spatial_error_regression_report.html")

        dep_idx = source.fields().lookupField(dep_var)
        exp_indices = [source.fields().lookupField(name) for name in indep_fields]
        if dep_idx < 0:
            raise QgsProcessingException(f"Dependent field '{dep_var}' not found.")
        missing = [name for name, idx in zip(indep_fields, exp_indices) if idx < 0]
        if missing:
            raise QgsProcessingException(f"Explanatory fields not found: {', '.join(missing)}")

        y_values = []
        x_rows = []
        valid_fids = []
        valid_dep_map = {}
        total = source.featureCount() or 1
        feedback.pushInfo("Extracting complete numeric records for spatial error regression...")
        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            dep_value = self._to_float(feature.attribute(dep_idx))
            exp_values = [self._to_float(feature.attribute(field_idx)) for field_idx in exp_indices]
            if dep_value is None or any(value is None for value in exp_values):
                continue
            fid = int(feature.id())
            y_values.append(dep_value)
            x_rows.append(exp_values)
            valid_fids.append(fid)
            valid_dep_map[fid] = dep_value
            feedback.setProgress(int(20 * (idx / total)))

        n = len(y_values)
        p = len(indep_fields)
        if n <= p + 2:
            raise QgsProcessingException(
                f"Insufficient complete records ({n}). Spatial Error Regression needs more observations "
                f"than explanatory variables, intercept, and spatial lambda ({p + 2} parameters)."
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
                "Increase the distance band, increase K, or review contiguity geometry before fitting a spatial error model."
            )
        if n_summary["isolated"] > 0:
            feedback.pushWarning(
                "The spatial error model includes isolated observations with no valid neighbors. "
                "Review the weight choice before using coefficients for planning decisions."
            )

        quality = regression_quality_summary(y, x_data, indep_fields, source.featureCount())
        for risk in quality["risks"]:
            feedback.pushWarning(risk)

        try:
            w = libpysal.weights.W(filtered_neighbors, filtered_weights, id_order=valid_id_order)
            w.transform = "r"
        except Exception as exc:
            raise QgsProcessingException(f"Could not build a libpysal W object for spatial error regression: {exc}")

        try:
            islands = list(getattr(w, "islands", []))
            if islands:
                feedback.pushWarning(f"{len(islands)} observation(s) are islands in the libpysal weights object.")
        except Exception:
            islands = []

        feedback.pushInfo("Fitting spatial error regression with spreg.ML_Error...")
        try:
            model = ml_error(
                y.reshape((-1, 1)),
                x_data,
                w=w,
                name_y=dep_var,
                name_x=list(indep_fields),
                name_w=weight_label,
            )
        except Exception as exc:
            raise QgsProcessingException(
                "Spatial error regression failed during model estimation. "
                "Review complete-record count, multicollinearity, weight density, and isolated observations. "
                f"Underlying error: {exc}"
            )

        results = self._extract_model_results(model, y, x_data, indep_fields)
        residual_std = float(np.std(results["residuals"]))
        if residual_std > 0:
            std_residuals = results["residuals"] / residual_std
        else:
            std_residuals = np.zeros(n)
        residual_spatial = residual_spatial_autocorrelation_summary(
            results["residuals"],
            filtered_neighbors,
            filtered_weights,
            valid_id_order,
        )
        push_residual_spatial_diagnostics(feedback, residual_spatial)

        feedback.pushInfo(
            "Spatial error model fitted: "
            f"lambda={format_number(results['lambda'], 6)}, "
            f"pseudo R2={format_number(results['pseudo_r2'], 6)}, "
            f"AIC={format_number(results['aic'], 4)}."
        )

        out_fields = source.fields()
        out_fields.append(QgsField("sem_pred", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("sem_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("sem_stdres", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("sem_nbrs", QVariant.Int))
        out_fields.append(QgsField("sem_used", QVariant.Int))

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
                "neighbors": len([nid for nid in filtered_neighbors.get(fid, []) if nid in valid_set]),
            }

        feedback.pushInfo("Writing spatial error regression audit fields...")
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            row = result_map.get(int(feature.id()))
            if row is None:
                out_feature.setAttribute("sem_pred", None)
                out_feature.setAttribute("sem_resid", None)
                out_feature.setAttribute("sem_stdres", None)
                out_feature.setAttribute("sem_nbrs", None)
                out_feature.setAttribute("sem_used", 0)
            else:
                out_feature.setAttribute("sem_pred", float(row["pred"]))
                out_feature.setAttribute("sem_resid", float(row["resid"]))
                out_feature.setAttribute("sem_stdres", float(row["stdres"]))
                out_feature.setAttribute("sem_nbrs", int(row["neighbors"]))
                out_feature.setAttribute("sem_used", 1)
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
            residual_spatial,
        )
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _load_spreg_dependencies(self):
        try:
            import libpysal
            from spreg import ML_Error
        except Exception as exc:
            raise QgsProcessingException(
                "Spatial Error Regression requires optional libraries libpysal and spreg. "
                "Run PlanX GeoStats Lab > 00 | Setup and Diagnostics > GeoStats Library Status "
                "to review the active QGIS Python environment, or Install / Update GeoStats "
                "Libraries to install with explicit approval. "
                f"Import error: {exc}"
            )
        return libpysal, ML_Error

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
            names.append("Spatial error lambda")
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
            "lambda": self._safe_float(getattr(model, "lam", betas[-1] if betas.size else None)),
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
                z_value = float(item[0])
                p_value = float(item[1])
            except (TypeError, ValueError, IndexError):
                z_value = np.nan
                p_value = np.nan
            if not (np.isfinite(z_value) and np.isfinite(p_value)):
                continue
            z_values[idx] = z_value
            p_values[idx] = p_value
        return z_values, p_values

    def _safe_float(self, value):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return result

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
        residual_spatial,
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

        lambda_value = results["lambda"]
        if lambda_value is None:
            lambda_text = "not reported"
        elif lambda_value > 0:
            lambda_text = "positive"
        elif lambda_value < 0:
            lambda_text = "negative"
        else:
            lambda_text = "near zero"

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
                "Review the spatial relationship before treating lambda as a stable planning signal.</div>"
            )

        if residual_spatial.get("available") and residual_spatial.get("p_value") is not None and residual_spatial["p_value"] < 0.05:
            next_action = (
                "Residuals still retain a spatial pattern after SEM. Compare the same specification in Spatial Lag, GWR, or MGWR, "
                "and review omitted variables or a different neighborhood definition."
            )
        else:
            next_action = (
                "Compare SEM against OLS and Spatial Lag using model fit, residual maps, and planning theory. "
                "Prefer SEM when spatial dependence appears to be in unobserved context or measurement error rather than direct spillover."
            )

        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Spatial Error Regression</title>
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
<h1>Spatial Error Regression (SEM)</h1>
<p class="subtitle">Dependent field: <strong>{html.escape(dep_var)}</strong> | Explanatory variables: <strong>{html.escape(', '.join(indep_fields))}</strong> | Weights: <strong>{html.escape(weight_detail)}</strong></p>

<section class="summary">
<strong>Executive summary.</strong> SEM estimates whether spatial structure remains in the unobserved part of the model after the explanatory variables are considered. A {html.escape(lambda_text)} lambda indicates the direction of spatial dependence in the error process. In planning analysis, this often points to missing contextual variables, boundary effects, measurement structure, or unmodeled spatial regimes rather than direct spillover in the dependent variable.
</section>

<div class="grid">
<div class="card"><div class="card-title">Complete Records</div><div class="card-value">{results['n']}</div></div>
<div class="card"><div class="card-title">Spatial Lambda</div><div class="card-value">{format_number(results['lambda'], 6)}</div></div>
<div class="card"><div class="card-title">Pseudo R2</div><div class="card-value">{format_number(results['pseudo_r2'], 6)}</div></div>
<div class="card"><div class="card-title">AIC</div><div class="card-value">{format_number(results['aic'], 4)}</div></div>
</div>

<h2>Method Assumptions</h2>
<p>The fitted model follows a spatial error structure where residual dependence is modeled through <strong>u = lambda W u + error</strong>. The explanatory variables remain global, while lambda absorbs spatial structure in the error process.</p>
<p>SEM is appropriate when the residual map suggests spatially structured omitted context or measurement process. It is not the same as a spatial lag model, which estimates direct neighborhood dependence in the dependent variable.</p>

{diagnostics_html(numeric_summary, n_summary, crs_warning)}
{regression_quality_html(quality)}
{residual_spatial_autocorrelation_html(residual_spatial)}
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
<p>A meaningful lambda suggests that residuals are spatially structured after the explanatory variables are included. This can improve inference compared with OLS when residual spatial autocorrelation would otherwise bias standard errors or hide omitted context. Interpret coefficient signs together with the residual diagnostic and the credibility of the selected spatial relationship.</p>

<h2>Recommended Analyst Action</h2>
<div class="next-action">{html.escape(next_action)}</div>

<h2>Caveats</h2>
<ul>
<li>SEM is sensitive to the chosen spatial weights. Contiguity, KNN, and distance-band weights represent different planning theories.</li>
<li>SEM does not estimate local coefficients. If relationships vary geographically, compare with GWR or MGWR.</li>
<li>Distance-band models should use a projected CRS with meaningful map units. Geographic degrees can produce misleading thresholds.</li>
<li>High multicollinearity, small sample size, islands, and unstable neighborhood support can make coefficients difficult to interpret.</li>
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

        feedback.pushInfo("Applying spatial error regression standardized residual styling...")
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
        layer.setRenderer(QgsGraduatedSymbolRenderer("sem_stdres", ranges))
        layer.triggerRepaint()
        return {}
