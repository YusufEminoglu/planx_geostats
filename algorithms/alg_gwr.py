# -*- coding: utf-8 -*-
"""Geographically Weighted Regression (GWR) Processing Algorithm."""
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
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFeatureSink,
    QgsProcessingOutputHtml,
    QgsFeatureSink
)

from ..core.stats_engines import calculate_gwr

logger = logging.getLogger("PlanX-GeoStats")


class GWRAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    DEP_VAR = "DEP_VAR"
    INDEPENDENTS = "INDEPENDENTS"
    KERNEL_TYPE = "KERNEL_TYPE"
    BANDWIDTH = "BANDWIDTH"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "gwr_regression"

    def displayName(self) -> str:
        return "Geographically Weighted Regression (GWR)"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def createInstance(self):
        return GWRAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Performs Geographically Weighted Regression (GWR), a local form of linear "
            "regression used to model spatially varying relationships.\n\n"
            "Calculates local regression equations at each feature location, producing "
            "spatially varying k-coefficients and standard errors. Outputs local diagnostics "
            "(residual, local R2, local t-stats) to the attributes of the output layer, "
            "and generates a detailed global diagnostics HTML report (AICc, Residual Sum of Squares, "
            "Effective Degrees of Freedom)."
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
        self.addParameter(
            QgsProcessingParameterField(
                self.INDEPENDENTS,
                "Independent variable fields",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.KERNEL_TYPE,
                "Kernel type",
                options=["Fixed Gaussian", "Fixed Bisquare", "Adaptive Bisquare"],
                defaultValue=2
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BANDWIDTH,
                "Bandwidth value (distance in map units for Fixed; number of neighbors for Adaptive)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=30.0,
                minValue=1.0
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output local coefficients layer"
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output GWR HTML report",
                fileFilter="HTML files (*.html)",
                optional=True
            )
        )
        self.addOutput(
            QgsProcessingOutputHtml(
                "HTML_REPORT_OUT",
                "GWR regression diagnostics report"
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

        kernel_idx = self.parameterAsEnum(parameters, self.KERNEL_TYPE, context)
        kernel_types = ["fixed_gaussian", "fixed_bisquare", "adaptive_bisquare"]
        kernel_type = kernel_types[kernel_idx]

        bandwidth = self.parameterAsDouble(parameters, self.BANDWIDTH, context)

        # Resolve output HTML file path
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, "gwr_report.html")

        # Map field indexes
        dep_idx = source.fields().lookupField(dep_var)
        indep_idxs = [source.fields().lookupField(name) for name in indep_fields]

        if dep_idx < 0:
            raise QgsProcessingException(f"Dependent field '{dep_var}' not found.")
        for name, idx in zip(indep_fields, indep_idxs):
            if idx < 0:
                raise QgsProcessingException(f"Independent field '{name}' not found.")

        feedback.pushInfo("Extracting numeric values, coordinates, and filtering missing data...")
        dep_vals = []
        indep_vals = []
        coords_list = []
        valid_fids = []

        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom = f.geometry()
            if geom.isEmpty():
                continue

            y_val = f.attribute(dep_idx)
            if y_val is None or y_val == QVariant() or str(y_val) == 'NULL':
                continue

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
                pt = geom.centroid().asPoint()
                coords_list.append([pt.x(), pt.y()])
                valid_fids.append(f.id())
            except (ValueError, TypeError):
                continue
            
            feedback.setProgress(int(20 * (idx / total)))

        n = len(dep_vals)
        p = len(indep_fields)

        if n <= p + 2:
            raise QgsProcessingException(
                f"Insufficient valid observations ({n}). "
                f"Must be greater than independent variables ({p}) + 2 for local regression."
            )

        y = np.array(dep_vals)
        X_data = np.array(indep_vals)
        coords = np.array(coords_list)

        feedback.pushInfo("Solving Geographically Weighted Regression...")
        results = calculate_gwr(y, X_data, coords, bandwidth, kernel_type)

        if feedback.isCanceled():
            return {}

        # Prepare output fields
        out_fields = source.fields()
        out_fields.append(QgsField("y_observed", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("y_predicted", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("residual", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("local_r2", QVariant.Double, len=10, prec=6))
        
        # Intercept and beta fields
        out_fields.append(QgsField("coef_int", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("se_int", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("t_int", QVariant.Double, len=10, prec=4))

        for name in indep_fields:
            # Keep names short to avoid shapefile field truncation (10 chars max)
            short_name = name[:5]
            out_fields.append(QgsField(f"coef_{short_name}", QVariant.Double, len=12, prec=6))
            out_fields.append(QgsField(f"se_{short_name}", QVariant.Double, len=12, prec=6))
            out_fields.append(QgsField(f"t_{short_name}", QVariant.Double, len=10, prec=4))

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

        # Map results
        local_beta = results["local_beta"]
        local_se = results["local_se"]
        local_t = results["local_t"]
        y_pred = results["y_pred"]
        residuals = results["residuals"]
        local_r2 = results["local_r2"]

        results_map = {}
        for idx, fid in enumerate(valid_fids):
            results_map[fid] = idx

        feedback.pushInfo("Writing local coefficients and local R2 to output...")
        for current, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            out_feat = QgsFeature(f)
            out_feat.setFields(out_fields)

            fid = f.id()
            if fid in results_map:
                idx = results_map[fid]
                out_feat.setAttribute("y_observed", float(y[idx]))
                out_feat.setAttribute("y_predicted", float(y_pred[idx]))
                out_feat.setAttribute("residual", float(residuals[idx]))
                out_feat.setAttribute("local_r2", float(local_r2[idx]))
                
                # Intercept
                out_feat.setAttribute("coef_int", float(local_beta[idx, 0]))
                out_feat.setAttribute("se_int", float(local_se[idx, 0]))
                out_feat.setAttribute("t_int", float(local_t[idx, 0]))

                # Independent coefficients
                for var_idx, name in enumerate(indep_fields):
                    short_name = name[:5]
                    out_feat.setAttribute(f"coef_{short_name}", float(local_beta[idx, var_idx + 1]))
                    out_feat.setAttribute(f"se_{short_name}", float(local_se[idx, var_idx + 1]))
                    out_feat.setAttribute(f"t_{short_name}", float(local_t[idx, var_idx + 1]))
            else:
                out_feat.setAttribute("y_observed", None)
                out_feat.setAttribute("y_predicted", None)
                out_feat.setAttribute("residual", None)
                out_feat.setAttribute("local_r2", None)
                out_feat.setAttribute("coef_int", None)
                out_feat.setAttribute("se_int", None)
                out_feat.setAttribute("t_int", None)
                for name in indep_fields:
                    short_name = name[:5]
                    out_feat.setAttribute(f"coef_{short_name}", None)
                    out_feat.setAttribute(f"se_{short_name}", None)
                    out_feat.setAttribute(f"t_{short_name}", None)

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 60 * (current / total)))

        # Generate GWR diagnostics report
        feedback.pushInfo("Generating global diagnostics HTML report...")
        self.write_html_report(results, html_path, dep_var, kernel_type, bandwidth)

        return {
            self.OUTPUT: dest_id,
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path
        }

    def write_html_report(self, res: dict, path: str, dep: str, kernel: str, bw: float):
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX-GeoStats GWR Diagnostics</title>
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
        <h1>Geographically Weighted Regression (GWR) Report</h1>
        <p class="subtitle">Dependent Variable: <strong>{dep}</strong> | Sample Size: <strong>{len(res["residuals"])}</strong></p>
    </header>

    <div class="grid">
        <div class="card">
            <div class="card-title">Global R-Squared</div>
            <div class="card-value">{res["r2"]:.6f}</div>
        </div>
        <div class="card">
            <div class="card-title">AICc Value</div>
            <div class="card-value">{res["aicc"]:.4f if res["aicc"] != np.inf else "N/A"}</div>
        </div>
        <div class="card">
            <div class="card-title">Residual Sum of Squares</div>
            <div class="card-value">{res["rss"]:.4f}</div>
        </div>
        <div class="card">
            <div class="card-title">Effective Degrees of Freedom</div>
            <div class="card-value">{res["effective_df"]:.4f}</div>
        </div>
    </div>

    <h2>Kernel Parameter Configurations</h2>
    <table>
        <thead>
            <tr>
                <th>Kernel Parameter</th>
                <th>Configured Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Spatial Kernel Selection</strong></td>
                <td>{kernel.replace('_', ' ').title()}</td>
            </tr>
            <tr>
                <td><strong>Bandwidth Settings</strong></td>
                <td>{bw} {"neighbors" if "adaptive" in kernel else "map units"}</td>
            </tr>
        </tbody>
    </table>

    <footer>
        Generated by PlanX-GeoStats local spatial statistics engine.
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

        feedback.pushInfo("Applying GWR local R-squared graduated styling...")

        # Graduated style on local_r2 showing localized model performance
        ranges = []
        range_definitions = [
            (0.0, 0.25, '#d7191c', 'Poor Local Fit (R2 < 0.25)'),
            (0.25, 0.50, '#fdae61', 'Moderate Local Fit (R2 0.25 to 0.50)'),
            (0.50, 0.75, '#ffffbf', 'Good Local Fit (R2 0.50 to 0.75)'),
            (0.75, 1.0, '#abdda4', 'Excellent Local Fit (R2 > 0.75)')
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

        renderer = QgsGraduatedSymbolRenderer('local_r2', ranges)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        return {}
