# -*- coding: utf-8 -*-
"""Ordinary Least Squares (OLS) Regression Processing Algorithm."""
from __future__ import annotations

import logging
import os
import tempfile
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsProject,
    QgsFeature,
    QgsField,
    QgsSymbol,
    QgsRendererRange,
    QgsGraduatedSymbolRenderer,
    QgsWkbTypes,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFeatureSink,
    QgsProcessingOutputHtml,
    QgsFeatureSink
)

from ..core.weights import build_weights_matrix
from ..core.stats_engines import calculate_ols

logger = logging.getLogger("PlanX GeoStats Lab")


class SpatialRegressionAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    DEP_VAR = "DEP_VAR"
    INDEPENDENTS = "INDEPENDENTS"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "ols_regression"

    def displayName(self) -> str:
        return "Ordinary Least Squares (OLS) Regression"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def createInstance(self):
        return SpatialRegressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Performs Ordinary Least Squares (OLS) linear regression analysis.\n\n"
            "Calculates relationships between a dependent variable and one or more independent variables. "
            "Outputs a residuals layer mapped by standard deviation of residuals, and generates a detailed "
            "diagnostic report containing R-squared, coefficients, Jarque-Bera, Koenker's Breusch-Pagan, "
            "and residual spatial autocorrelation diagnostics."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.DEP_VAR,
                "Dependent variable field",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )
        # We allow multiple independent variable fields
        self.addParameter(
            QgsProcessingParameterField(
                self.INDEPENDENTS,
                "Independent variable fields (select one or more)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output residuals layer"
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output OLS HTML report",
                fileFilter="HTML files (*.html)",
                optional=True
            )
        )
        # Register QgsProcessingOutputHtml so QGIS Results Viewer displays the HTML
        self.addOutput(
            QgsProcessingOutputHtml(
                "HTML_REPORT_OUT",
                "OLS regression diagnostic report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        dep_var = self.parameterAsString(parameters, self.DEP_VAR, context)
        indep_fields = self.parameterAsFields(parameters, self.INDEPENDENTS, context)

        if not indep_fields:
            raise QgsProcessingException("At least one independent variable must be selected.")

        # Resolve output HTML file path
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, "ols_regression_report.html")

        # Map field indexes
        dep_idx = source.fields().lookupField(dep_var)
        indep_idxs = [source.fields().lookupField(name) for name in indep_fields]

        # Check for invalid indexes
        if dep_idx < 0:
            raise QgsProcessingException(f"Dependent field '{dep_var}' not found.")
        for name, idx in zip(indep_fields, indep_idxs):
            if idx < 0:
                raise QgsProcessingException(f"Independent field '{name}' not found.")

        feedback.pushInfo("Extracting numeric values and filtering missing data...")
        
        dep_vals = []
        indep_vals = []
        valid_fids = []

        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            # Dependent value
            y_val = f.attribute(dep_idx)
            if y_val is None or y_val == QVariant() or str(y_val) == 'NULL':
                continue

            # Independent values
            has_null = False
            f_indeps = []
            for i_idx in indep_idxs:
                x_val = f.attribute(i_idx)
                if x_val is None or x_val == QVariant() or str(x_val) == 'NULL':
                    has_null = True
                    break
                try:
                    f_indeps.append(float(x_val))
                except (ValueError, TypeError):
                    has_null = True
                    break

            if has_null:
                continue

            try:
                dep_vals.append(float(y_val))
                indep_vals.append(f_indeps)
                valid_fids.append(f.id())
            except (ValueError, TypeError):
                continue
            
            feedback.setProgress(int(20 * (idx / total)))

        n = len(dep_vals)
        p = len(indep_fields)

        if n <= p + 1:
            raise QgsProcessingException(
                f"Insufficent valid observations ({n}). "
                f"Must be greater than the number of independent variables ({p}) + 1."
            )

        y = np.array(dep_vals)
        X_data = np.array(indep_vals)

        # Build weights matrix dynamically for spatial autocorrelation diagnostics of residuals
        geom_type = source.geometryType()
        res_weight_type = "queen" if geom_type == QgsWkbTypes.PolygonGeometry else "knn"
        feedback.pushInfo(f"Building {res_weight_type} weights matrix for residual spatial autocorrelation test...")
        
        neighbors, weights, id_order, _ = build_weights_matrix(
            source,
            res_weight_type,
            k_neighbors=8,
            feedback=feedback
        )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Running OLS regression analysis...")
        results = calculate_ols(y, X_data, neighbors, weights, valid_fids, indep_fields)

        # Prepare output fields
        out_fields = source.fields()
        out_fields.append(QgsField("residual", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("std_res", QVariant.Double, len=12, prec=6))

        # Setup sink
        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs()
        )
        self.out_layer_id = dest_id

        # Write results
        residuals = results["residuals"]
        std_residuals = results["std_residuals"]
        results_map = {fid: (residuals[i], std_residuals[i]) for i, fid in enumerate(valid_fids)}

        feedback.pushInfo("Writing residual attributes...")
        for current, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            out_feat = QgsFeature(f)
            out_feat.setFields(out_fields)

            fid = f.id()
            if fid in results_map:
                res, std_res = results_map[fid]
                out_feat.setAttribute("residual", float(res))
                out_feat.setAttribute("std_res", float(std_res))
            else:
                out_feat.setAttribute("residual", None)
                out_feat.setAttribute("std_res", None)

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 30 * (current / total)))

        # Generate HTML report
        feedback.pushInfo("Generating regression HTML diagnostics report...")
        self.write_html_report(results, html_path, dep_var)

        # Return results including the HTML output mapping
        return {
            self.OUTPUT: dest_id,
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path
        }

    def write_html_report(self, res: dict, path: str, dep_var: str):
        # Format diagnostics badges
        jb_stat, jb_p = res["jarque_bera"]
        bp_stat, bp_p = res["breusch_pagan"]
        moran_i = res["residuals_moran"]

        jb_badge = '<span class="badge badge-success">Normally Distributed</span>' if jb_p >= 0.05 else '<span class="badge badge-danger">Non-Normal Residuals</span>'
        bp_badge = '<span class="badge badge-success">Homoskedastic</span>' if bp_p >= 0.05 else '<span class="badge badge-warning">Heteroskedastic</span>'
        
        moran_status = "No Significant Autocorrelation"
        moran_class = "badge-success"
        if abs(moran_i) > 0.15: # Proxy threshold
            moran_status = "Autocorrelated Residuals"
            moran_class = "badge-danger"
        moran_badge = f'<span class="badge {moran_class}">{moran_status}</span>'

        # Build coefficient rows
        coef_rows = ""
        for i, name in enumerate(res["variable_names"]):
            coeff = res["coefficients"][i]
            se = res["std_errors"][i]
            t_stat = res["t_statistics"][i]
            p_val = res["p_values"][i]
            
            p_formatted = f"{p_val:.6f}" if p_val >= 0.0001 else "< 0.0001"
            p_class = "significant" if p_val < 0.05 else "non-significant"

            coef_rows += f"""
            <tr class="{p_class}">
                <td><strong>{name}</strong></td>
                <td>{coeff:.6f}</td>
                <td>{se:.6f}</td>
                <td>{t_stat:.4f}</td>
                <td>{p_formatted}</td>
            </tr>"""

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab OLS Regression Diagnostics</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #2d3748;
        background-color: #f7fafc;
        margin: 0;
        padding: 20px;
        line-height: 1.5;
    }}
    .container {{
        max-width: 860px;
        margin: 0 auto;
        background: #ffffff;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }}
    header {{
        border-bottom: 2px solid #edf2f7;
        padding-bottom: 20px;
        margin-bottom: 25px;
    }}
    h1 {{
        color: #1a202c;
        margin: 0 0 5px 0;
        font-size: 1.75rem;
    }}
    .subtitle {{
        color: #718096;
        margin: 0;
        font-size: 0.95rem;
    }}
    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 15px;
        margin-bottom: 30px;
    }}
    .card {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
    }}
    .card-title {{
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #4a5568;
        margin-bottom: 5px;
        font-weight: 700;
        letter-spacing: 0.05em;
    }}
    .card-value {{
        font-size: 1.4rem;
        font-weight: 800;
        color: #2b6cb0;
    }}
    h2 {{
        font-size: 1.2rem;
        color: #2d3748;
        margin-top: 30px;
        margin-bottom: 15px;
        border-left: 4px solid #3182ce;
        padding-left: 10px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 25px;
    }}
    th, td {{
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid #edf2f7;
        font-size: 0.9rem;
    }}
    th {{
        background-color: #ebf8ff;
        color: #2b6cb0;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }}
    .significant {{
        background-color: #f0fff4;
    }}
    .non-significant {{
        color: #718096;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 4px;
    }}
    .badge-success {{ background-color: #c6f6d5; color: #22543d; }}
    .badge-warning {{ background-color: #feebc8; color: #744210; }}
    .badge-danger {{ background-color: #fed7d7; color: #742a2a; }}
    footer {{
        margin-top: 40px;
        border-top: 1px solid #edf2f7;
        padding-top: 15px;
        font-size: 0.8rem;
        color: #a0aec0;
        text-align: center;
    }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Ordinary Least Squares (OLS) Regression Report</h1>
        <p class="subtitle">Dependent Variable: <strong>{dep_var}</strong> | Analysis sample size: <strong>{res["n"]}</strong></p>
    </header>

    <div class="grid">
        <div class="card">
            <div class="card-title">R-Squared</div>
            <div class="card-value">{res["r2"]:.6f}</div>
        </div>
        <div class="card">
            <div class="card-title">Adjusted R-Squared</div>
            <div class="card-value">{res["adj_r2"]:.6f}</div>
        </div>
        <div class="card">
            <div class="card-title">Residual Std Error</div>
            <div class="card-value">{math.sqrt(res["residuals"].dot(res["residuals"]) / res["df_err"]):.6f}</div>
        </div>
        <div class="card">
            <div class="card-title">Residual DF</div>
            <div class="card-value">{res["df_err"]}</div>
        </div>
    </div>

    <h2>Variable Estimates</h2>
    <table>
        <thead>
            <tr>
                <th>Variable Name</th>
                <th>Coefficient</th>
                <th>Std Error</th>
                <th>t-Statistic</th>
                <th>Probability (p-value)</th>
            </tr>
        </thead>
        <tbody>
            {coef_rows}
        </tbody>
    </table>

    <h2>Model Diagnostics</h2>
    <table>
        <thead>
            <tr>
                <th>Diagnostic Test</th>
                <th>Statistic</th>
                <th>p-value / Indication</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Jarque-Bera Test</strong> (Normality of Residuals)</td>
                <td>{jb_stat:.4f}</td>
                <td>{jb_p:.6f}</td>
                <td>{jb_badge}</td>
            </tr>
            <tr>
                <td><strong>Koenker's Breusch-Pagan Test</strong> (Heteroskedasticity)</td>
                <td>{bp_stat:.4f}</td>
                <td>{bp_p:.6f}</td>
                <td>{bp_badge}</td>
            </tr>
            <tr>
                <td><strong>Moran's I on Residuals</strong> (Spatial Autocorrelation)</td>
                <td>{moran_i:.4f}</td>
                <td>N/A (Spatial Autocorrelation index)</td>
                <td>{moran_badge}</td>
            </tr>
        </tbody>
    </table>

    <footer>
        Generated by PlanX GeoStats Lab spatial statistics engine.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}

        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}

        feedback.pushInfo("Applying OLS standardized residuals graduated styling...")
        
        # 7-class diverging standard deviation classes
        ranges = []
        range_definitions = [
            (-9999.0, -2.5, '#2166ac', '< -2.5 Std Dev (Underprediction)'),
            (-2.5, -1.5, '#67a9cf', '-2.5 to -1.5 Std Dev'),
            (-1.5, -0.5, '#d1e5f0', '-1.5 to -0.5 Std Dev'),
            (-0.5, 0.5, '#f7f7f7', '-0.5 to 0.5 Std Dev (Near Zero)'),
            (0.5, 1.5, '#fddbc7', '0.5 to 1.5 Std Dev'),
            (1.5, 2.5, '#f4a582', '1.5 to 2.5 Std Dev'),
            (2.5, 9999.0, '#b2182b', '> 2.5 Std Dev (Overprediction)')
        ]

        for min_v, max_v, color_hex, label in range_definitions:
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(QColor(color_hex))
            symbol.setOpacity(0.85)

            if symbol.symbolLayerCount() > 0:
                sl = symbol.symbolLayer(0)
                if hasattr(sl, 'setStrokeColor'):
                    sl.setStrokeColor(QColor('#b0b0b0'))
                if hasattr(sl, 'setStrokeWidth'):
                    sl.setStrokeWidth(0.1)
                if hasattr(sl, 'setOutlineColor'):
                    sl.setOutlineColor(QColor('#b0b0b0'))

            r_range = QgsRendererRange(min_v, max_v, symbol, label)
            ranges.append(r_range)

        renderer = QgsGraduatedSymbolRenderer('std_res', ranges)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        return {}
