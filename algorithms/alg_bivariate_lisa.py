# -*- coding: utf-8 -*-
"""Bivariate Local Moran's I (Bivariate LISA) Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile
import numpy as np

from qgis.PyQt.QtCore import QVariant
from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsFeature,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFileDestination,
    QgsFeatureSink
)

from ..core.weights import build_weights_matrix
from ..core.stats_engines import calculate_bivariate_local_moran
from ..core.layer_metadata import apply_output_metadata
from ..core.local_pattern_audit import local_moran_class_summary
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, donut_chart_svg, kpi_card_row_html
from ..core.symbology import LISA_QUADRANT_STYLE, apply_renderer, lisa_quadrant_renderer

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class BivariateLISAAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD_X = "FIELD_X"
    FIELD_Y = "FIELD_Y"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    PERMUTATIONS = "PERMUTATIONS"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "bivariate_lisa"

    def displayName(self) -> str:
        return "Bivariate Cluster and Outlier Analysis (Bivariate LISA)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def icon(self):
        return algorithm_icon("bivariate_lisa")

    def createInstance(self):
        return BivariateLISAAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Extends Local Moran's I to two variables: for each feature, tests "
            "whether variable X at that location is significantly associated "
            "with the spatial lag (neighborhood average) of variable Y, using "
            "conditional-permutation inference (999 permutations by default, "
            "configurable 99-9999) rather than an analytical formula - the "
            "bivariate case has no simple closed-form null distribution.\n\n"
            "Output fields: bilisa_i, bilisa_z, bilisa_p, bilisa_nb (valid "
            "neighbor count), and quadrant, assigned when bilisa_p < 0.05:\n"
            "- HH (High-High): high X surrounded by high neighboring Y\n"
            "- LL (Low-Low): low X surrounded by low neighboring Y\n"
            "- HL (High-Low): high X surrounded by low neighboring Y\n"
            "- LH (Low-High): low X surrounded by high neighboring Y\n\n"
            "Bivariate LISA is directional and asymmetric: X's local relationship "
            "to lagged Y is not the same computation as Y's local relationship to "
            "lagged X, so running the tool with the fields swapped can produce a "
            "genuinely different map, not just a relabeled one - run both "
            "directions when the causal direction between X and Y is unclear. As "
            "with any bivariate spatial statistic, a significant result reflects "
            "association, not causation, and can be inflated by each variable's "
            "own spatial autocorrelation even when no real cross-variable "
            "relationship exists; check univariate Local Moran's I on X and Y "
            "separately as a sanity check."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.SourceType.TypeVectorAnyGeometry]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_X,
                "First numeric field (Variable X)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.DataType.Numeric
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_Y,
                "Second numeric field (Variable Y)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.DataType.Numeric
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial relationship / weights type",
                options=["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of neighbors (K value, KNN only)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=5,
                minValue=1
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                "Distance band threshold (map units, Distance Band only)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=1000.0,
                minValue=0.0001
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PERMUTATIONS,
                "Number of permutations (Monte Carlo)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=999,
                minValue=99,
                maxValue=9999
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Bivariate LISA Output Layer"
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output HTML report",
                fileFilter="HTML files (*.html)",
                optional=True
            )
        )
        self.addOutput(
            QgsProcessingOutputHtml(
                "HTML_REPORT_OUT",
                "Bivariate LISA cluster and outlier classification report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_bivariate_lisa_report.html")

        field_x_name = self.parameterAsString(parameters, self.FIELD_X, context)
        field_y_name = self.parameterAsString(parameters, self.FIELD_Y, context)
        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]

        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        perms = self.parameterAsInt(parameters, self.PERMUTATIONS, context)

        # Validate target fields
        field_x_idx = source.fields().lookupField(field_x_name)
        if field_x_idx < 0:
            raise QgsProcessingException(f"Variable X field '{field_x_name}' not found.")
        field_x = source.fields().at(field_x_idx)
        if not field_x.isNumeric():
            raise QgsProcessingException(f"Variable X field '{field_x_name}' must be numeric.")

        field_y_idx = source.fields().lookupField(field_y_name)
        if field_y_idx < 0:
            raise QgsProcessingException(f"Variable Y field '{field_y_name}' not found.")
        field_y = source.fields().at(field_y_idx)
        if not field_y.isNumeric():
            raise QgsProcessingException(f"Variable Y field '{field_y_name}' must be numeric.")

        feedback.pushInfo("Generating spatial weights matrix...")
        neighbors, weights, id_order, _ = build_weights_matrix(
            source,
            weight_type,
            k_neighbors=k_neighbors,
            distance_band=distance_band,
            feedback=feedback
        )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Extracting target field values...")
        x_dict = {}
        y_dict = {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            val_x = f.attribute(field_x_name)
            val_y = f.attribute(field_y_name)
            if val_x in (None, NULL) or val_y in (None, NULL):
                continue
            try:
                x_dict[f.id()] = float(val_x)
                y_dict[f.id()] = float(val_y)
            except (ValueError, TypeError):
                continue

        # Filter id_order and construct x, y arrays
        valid_id_order = [fid for fid in id_order if fid in x_dict and fid in y_dict]
        x_arr = np.array([x_dict[fid] for fid in valid_id_order])
        y_arr = np.array([y_dict[fid] for fid in valid_id_order])

        if len(x_arr) <= 2:
            raise QgsProcessingException("At least 3 valid features with numeric values are required for LISA analysis.")

        feedback.pushInfo(f"Calculating Bivariate Local Moran's I statistics using {perms} permutations...")
        i_vals, z_scores, p_values, quadrants = calculate_bivariate_local_moran(
            x_arr,
            y_arr,
            neighbors,
            weights,
            valid_id_order,
            permutations=perms
        )
        class_summary = local_moran_class_summary(quadrants)
        feedback.pushInfo(class_summary["message"])

        if feedback.isCanceled():
            return {}

        # Prepare output fields
        out_fields = source.fields()
        out_fields.append(QgsField("bilisa_i", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("bilisa_z", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("bilisa_p", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("quadrant", QVariant.String, len=20))
        out_fields.append(QgsField("bilisa_nb", QVariant.Int))

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

        # Map results to feature IDs
        results_map = {}
        valid_ids = set(valid_id_order)
        isolated_count = 0
        for idx, fid in enumerate(valid_id_order):
            neighbor_count = len([nid for nid in neighbors.get(fid, []) if nid in valid_ids])
            if neighbor_count == 0:
                isolated_count += 1
            results_map[fid] = (i_vals[idx], z_scores[idx], p_values[idx], quadrants[idx], neighbor_count)
        if isolated_count:
            feedback.pushWarning(
                f"{isolated_count} feature(s) had no valid neighbors. Consider a larger distance band or K value."
            )

        feedback.pushInfo("Writing results to output layer...")
        total = source.featureCount() or 1
        for current, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            out_feat = QgsFeature(f)
            out_feat.setFields(out_fields)

            fid = f.id()
            if fid in results_map:
                i_val, z, p, quad, neighbor_count = results_map[fid]
                out_feat.setAttribute("bilisa_i", float(i_val))
                out_feat.setAttribute("bilisa_z", float(z))
                out_feat.setAttribute("bilisa_p", float(p))
                out_feat.setAttribute("quadrant", str(quad))
                out_feat.setAttribute("bilisa_nb", int(neighbor_count))
            else:
                out_feat.setAttribute("bilisa_i", None)
                out_feat.setAttribute("bilisa_z", None)
                out_feat.setAttribute("bilisa_p", None)
                out_feat.setAttribute("quadrant", "Not Significant")
                out_feat.setAttribute("bilisa_nb", None)

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(50 + 50 * (current / total)))

        feedback.pushInfo("Generating HTML report...")
        self._write_html(html_path, field_x_name, field_y_name, len(valid_id_order), weight_type, class_summary, isolated_count)

        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, field_x_name, field_y_name, n, weight_type, class_summary, isolated_count):
        counts = class_summary["counts"]
        donut_labels = [label for _, _, label in LISA_QUADRANT_STYLE]
        donut_values = [counts.get(code, 0) for code, _, _ in LISA_QUADRANT_STYLE]
        donut_colors = {label: color for code, color, label in LISA_QUADRANT_STYLE}

        kpi_row = kpi_card_row_html([
            {"label": "Cluster features", "value": str(class_summary["cluster_count"]), "sublabel": "HH + LL", "tone": "good" if class_summary["cluster_count"] else "neutral"},
            {"label": "Outlier features", "value": str(class_summary["outlier_count"]), "sublabel": "HL + LH", "tone": "warn" if class_summary["outlier_count"] else "neutral"},
            {"label": "Significant total", "value": f"{class_summary['significant_count']} / {n}", "sublabel": f"Dominant: {class_summary['dominant_label']}"},
        ])

        guidance_html = analyst_guidance_html(
            "Bivariate LISA",
            "Bivariate Local Moran's I tests whether a feature's value on Variable X is spatially associated with its neighbors' values on a DIFFERENT variable Y, using the same HH/LL/HL/LH quadrant classification as univariate Local Moran's I.",
            [
                "bilisa_nb is checked before trusting a 'Not Significant' result - zero valid neighbors looks identical to a genuine null result.",
                "X and Y are conceptually distinct variables, not the same field measured twice (that case is univariate Local Moran's I).",
                "A significant class forms a spatially coherent group with its neighbors, not an isolated single cell.",
            ],
            [
                f"{isolated_count} feature(s) had zero valid neighbors" if isolated_count else "No isolated (zero-neighbor) features were found.",
                "Bivariate LISA does not control for the ordinary (non-spatial) correlation between X and Y - a strong global correlation can itself produce apparent spatial association.",
                "Each feature tested independently at p < 0.05 with no multiple-testing correction.",
            ],
            [
                "Local Moran's I (univariate) as a baseline for each field separately",
                "Global Lee's L for a whole-study-area bivariate spatial association summary before drilling into local results",
                "OLS/GWR if the goal is to model the X-Y relationship rather than just map where it clusters",
            ],
            "Use HH/LL bivariate clusters to identify areas where two different planning variables reinforce each other spatially; HL/LH outliers flag areas where they diverge, which may warrant closer inspection.",
        )

        donut_html = donut_chart_svg(donut_labels, donut_values, colors=donut_colors, title="Bivariate LISA Class Breakdown")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Bivariate LISA Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 760px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }}
    header {{ border-bottom: 2px solid #edf2f7; padding-bottom: 20px; margin-bottom: 25px; }}
    h1 {{ color: #1a202c; margin: 0 0 5px 0; font-size: 1.6rem; }}
    .subtitle {{ color: #718096; margin: 0; font-size: 0.95rem; }}
    section {{ margin: 28px 0; }}
    h2 {{ color: #1a202c; font-size: 1.15rem; margin: 0 0 12px 0; }}
    footer {{ margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 15px; font-size: 0.8rem; color: #a0aec0; text-align: center; }}
    {analyst_guidance_css()}
    {chart_css()}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Bivariate LISA (Bivariate Local Moran's I)</h1>
        <p class="subtitle">Variable X: <strong>{html.escape(field_x_name)}</strong> | Variable Y: <strong>{html.escape(field_y_name)}</strong> | Feature Count: <strong>{n}</strong> | Weights: <strong>{html.escape(weight_type)}</strong></p>
    </header>

    {kpi_row}

    <section>
        <h2>Bivariate LISA Class Breakdown</h2>
        {donut_html}
        <p class="chart-caption">{html.escape(class_summary["message"])}</p>
    </section>

    {guidance_html}

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

        layer = context.getMapLayer(self.out_layer_id)
        if not layer:
            return {}

        feedback.pushInfo("Applying LISA cluster analysis styling...")
        apply_output_metadata(
            layer,
            "PlanX GeoStats Bivariate Local Moran cluster and outlier output",
            {
                "bilisa_i": "Bivariate Local Moran's I statistic",
                "bilisa_z": "Bivariate Local Moran z-score",
                "bilisa_p": "Bivariate Local Moran p-value",
                "quadrant": "Bivariate LISA class: HH, LL, HL, LH, or Not Significant",
                "bilisa_nb": "Valid neighbors used for the local statistic",
            },
            self.displayName(),
        )
        apply_renderer(layer, lisa_quadrant_renderer(layer.geometryType(), "quadrant"))

        return {}
