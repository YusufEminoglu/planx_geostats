# -*- coding: utf-8 -*-
"""Getis-Ord Gi* Hotspot Analysis Processing Algorithm."""
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
    QgsProject,
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
from ..core.stats_engines import calculate_getis_ord
from ..core.layer_metadata import apply_output_metadata
from ..core.local_pattern_audit import getis_ord_class_summary
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, donut_chart_svg, kpi_card_row_html
from ..core.symbology import GI_CONFIDENCE_STYLE, apply_renderer, gi_confidence_renderer

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class GetisOrdAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "getis_ord_gi"

    def displayName(self) -> str:
        return "Hot Spot Analysis (Getis-Ord Gi*)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def icon(self):
        return algorithm_icon("getis_ord_gi")

    def createInstance(self):
        return GetisOrdAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes the Getis-Ord Gi* statistic for every feature: whether a "
            "feature and its neighbors together form a statistically significant "
            "hot spot (high values), cold spot (low values), or neither, relative "
            "to the study area as a whole. Gi* always includes the feature itself "
            "in its own neighborhood sum (unlike plain Gi), which behaves better "
            "at the edges of the study area and is the standard variant used in "
            "practice.\n\n"
            "Output fields: gi_zscore, gi_pvalue, gi_conf (a signed confidence "
            "bin: +/-3 at p < 0.01, +/-2 at p < 0.05, +/-1 at p < 0.10, 0 "
            "otherwise; sign gives hot vs cold), and gi_nbrs (how many valid "
            "neighbors supported the local statistic). Regardless of the chosen "
            "weight type (Queen, Rook, KNN, Distance Band), every neighbor "
            "contributes equally - the weight type only decides who counts as a "
            "neighbor, not how strongly; there is no distance-decay weighting "
            "inside Gi* itself.\n\n"
            "A feature with zero valid neighbors is silently assigned z = 0, p = "
            "1, conf = 0 - it looks exactly like 'not significant' in the output "
            "even though the real issue is missing neighborhood support, not an "
            "absence of pattern. Always cross-check gi_nbrs before trusting a "
            "'not significant' result, especially with sparse KNN or small "
            "distance bands. Gi* finds clusters, not outliers - where a high "
            "value surrounded by low neighbors (or vice versa) is the object of "
            "interest, use Local Moran's I instead."
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
                self.FIELD,
                "Target numeric field to analyze",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
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
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                "Distance band threshold (map units, Distance Band only)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                minValue=0.0001
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Hot Spot analysis output layer"
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
                "Hot spot classification report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Retrieve parameters
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_getis_ord_report.html")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]

        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)

        # Validate target field
        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Target field '{field_name}' not found.")

        field = source.fields().at(field_idx)
        if not field.isNumeric():
            raise QgsProcessingException(f"Target field '{field_name}' must be numeric.")

        feedback.pushInfo("Generating spatial weights matrix...")

        # Build weights matrix
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
        y_dict = {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            val = f.attribute(field_name)
            if val is None or val == NULL or str(val) == 'NULL':
                continue
            try:
                y_dict[f.id()] = float(val)
            except (ValueError, TypeError):
                continue

        # Filter id_order and construct y array
        valid_id_order = [fid for fid in id_order if fid in y_dict]
        y = np.array([y_dict[fid] for fid in valid_id_order])

        if len(y) == 0:
            raise QgsProcessingException("No valid numeric values found in the target field.")

        feedback.pushInfo("Calculating Getis-Ord Gi* statistics...")
        z_scores, p_values, conf_bins = calculate_getis_ord(
            y,
            neighbors,
            weights,
            valid_id_order,
            star=True
        )
        class_summary = getis_ord_class_summary(conf_bins)
        feedback.pushInfo(class_summary["message"])

        if feedback.isCanceled():
            return {}

        # Prepare output fields
        out_fields = source.fields()
        out_fields.append(QgsField("gi_zscore", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("gi_pvalue", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("gi_conf", QVariant.Int))
        out_fields.append(QgsField("gi_nbrs", QVariant.Int))

        # Initialize output sink
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
            results_map[fid] = (z_scores[idx], p_values[idx], conf_bins[idx], neighbor_count)
        if isolated_count:
            feedback.pushWarning(
                f"{isolated_count} feature(s) had no valid neighbors. Review gi_nbrs and consider a larger distance band or K value."
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
                z, p, c, neighbor_count = results_map[fid]
                out_feat.setAttribute("gi_zscore", float(z))
                out_feat.setAttribute("gi_pvalue", float(p))
                out_feat.setAttribute("gi_conf", int(c))
                out_feat.setAttribute("gi_nbrs", int(neighbor_count))
            else:
                out_feat.setAttribute("gi_zscore", None)
                out_feat.setAttribute("gi_pvalue", None)
                out_feat.setAttribute("gi_conf", None)
                out_feat.setAttribute("gi_nbrs", None)

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(50 + 50 * (current / total)))

        feedback.pushInfo("Generating HTML report...")
        self._write_html(html_path, field_name, len(y), weight_type, class_summary, isolated_count)

        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, field_name, n, weight_type, class_summary, isolated_count):
        counts = class_summary["counts"]
        donut_labels = [label for _, _, label in GI_CONFIDENCE_STYLE]
        donut_values = [counts.get(val, 0) for val, _, _ in GI_CONFIDENCE_STYLE]
        donut_colors = {label: color for val, color, label in GI_CONFIDENCE_STYLE}

        kpi_row = kpi_card_row_html([
            {"label": "Hot spot features", "value": str(class_summary["hot_count"]), "sublabel": "gi_conf > 0", "tone": "warn" if class_summary["hot_count"] else "neutral"},
            {"label": "Cold spot features", "value": str(class_summary["cold_count"]), "sublabel": "gi_conf < 0", "tone": "good" if class_summary["cold_count"] else "neutral"},
            {"label": "Significant total", "value": f"{class_summary['significant_count']} / {n}", "sublabel": f"Dominant: {class_summary['dominant_label']}"},
        ])

        guidance_html = analyst_guidance_html(
            "Getis-Ord Gi* Hot Spot Analysis",
            "Gi* tests whether each feature and its neighbors together form a statistically significant concentration of high (hot) or low (cold) values relative to the whole study area.",
            [
                "gi_nbrs is checked before trusting a 'Not Significant' result - zero valid neighbors looks identical to a genuine null result.",
                "The weight type (Queen/Rook/KNN/Distance Band) is a defensible neighborhood definition for this layer.",
                "A hot or cold spot forms a spatially coherent group rather than an isolated significant cell.",
            ],
            [
                f"{isolated_count} feature(s) had zero valid neighbors" if isolated_count else "No isolated (zero-neighbor) features were found.",
                "A significant class appearing as a single isolated cell rather than a coherent cluster.",
                "Each feature tested independently at p < 0.05/0.10/0.01 with no multiple-testing correction.",
            ],
            [
                "Local Moran's I (LISA) for cluster AND outlier detection (HL/LH), not just hot/cold magnitude",
                "Incremental Spatial Autocorrelation to pick a defensible neighborhood scale first",
                "Spatial Gini / Colocation Quotient to quantify the concentration Gi* is flagging",
            ],
            "Use hot/cold spot classes to prioritize field verification or targeted intervention areas, not as a standalone causal claim about why a concentration exists.",
        )

        kpi_row_html = kpi_row
        donut_html = donut_chart_svg(donut_labels, donut_values, colors=donut_colors, title="Hot/Cold Spot Class Breakdown")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Hot Spot Analysis Report</title>
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
        <h1>Hot Spot Analysis (Getis-Ord Gi*)</h1>
        <p class="subtitle">Field Analyzed: <strong>{html.escape(field_name)}</strong> | Feature Count: <strong>{n}</strong> | Weights: <strong>{html.escape(weight_type)}</strong></p>
    </header>

    {kpi_row_html}

    <section>
        <h2>Hot/Cold Spot Class Breakdown</h2>
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
        # Applies gorgeous hot/cold styling automatically in the QGIS GUI thread
        if self.out_layer_id is None:
            return {}

        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}

        feedback.pushInfo("Applying Cold-to-Hot Hotspot symbology style...")
        apply_output_metadata(
            layer,
            "PlanX GeoStats Getis-Ord Gi* hot spot output",
            {
                "gi_zscore": "Getis-Ord Gi* z-score",
                "gi_pvalue": "Getis-Ord Gi* p-value",
                "gi_conf": "Hot/cold confidence class from -3 cold to +3 hot",
                "gi_nbrs": "Valid neighbors used for the local statistic",
            },
            self.displayName(),
        )

        apply_renderer(layer, gi_confidence_renderer(layer.geometryType(), "gi_conf"))

        return {}
