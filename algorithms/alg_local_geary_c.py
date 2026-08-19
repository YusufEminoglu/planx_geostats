# -*- coding: utf-8 -*-
"""Local Geary's C (Cluster and Outlier Analysis) Processing Algorithm."""
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
    QgsFeatureSink,
)

from ..core.weights import build_weights_matrix
from ..core.advanced_stats_engines import calculate_local_geary_c
from ..core.layer_metadata import apply_output_metadata
from ..core.local_pattern_audit import local_moran_class_summary
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, donut_chart_svg, kpi_card_row_html
from ..core.symbology import LISA_QUADRANT_STYLE, apply_renderer, lisa_quadrant_renderer

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class LocalGearyCAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "local_geary_c"

    def displayName(self) -> str:
        return "Cluster and Outlier Analysis (Local Geary's C)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def icon(self):
        return algorithm_icon("local_geary_c")

    def createInstance(self):
        return LocalGearyCAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes Anselin's (2019) Local Geary's C for every feature - the "
            "local, per-feature complement to the plugin's global Geary's C "
            "(Group 02), sensitive to sharp local dissimilarity rather than "
            "deviation from the study-area mean. Output fields: lgc_c (local "
            "Geary's C, sum of neighbor squared differences), lgc_z, lgc_p "
            "(both from conditional permutation, not an analytical formula), "
            "lgc_nbrs, and quadrant, assigned only when lgc_p < 0.05:\n"
            "- HH/LL: the feature and its neighbors are unusually SIMILAR to "
            "each other and jointly high (HH) or low (LL) - a local cluster\n"
            "- HL/LH: the feature and its neighbors are unusually DISSIMILAR - "
            "a high value among low neighbors (HL) or the reverse (LH)\n"
            "- Not Significant: p >= 0.05, or the feature had no valid neighbors\n\n"
            "Uses the same HH/LL/HL/LH vocabulary as Local Moran's I so results "
            "are directly comparable, but the underlying test is different: "
            "Local Moran compares each value's deviation from the mean against "
            "its neighbors' deviations, while Local Geary compares each pair of "
            "neighboring values directly - it can flag a location as an outlier "
            "purely from sharp local contrast even where Local Moran sees "
            "nothing unusual relative to the global mean. Significance uses "
            "conditional permutation (999 by default): each feature's own "
            "value is held fixed while its neighbor set is resampled from the "
            "rest of the layer, matching the plugin's Bivariate LISA "
            "convention rather than a closed-form variance formula. Power is "
            "genuinely limited on small or near-binary fields - a sharp "
            "two-value boundary can show as 'Not Significant' simply because "
            "most of a boundary feature's neighbors are still on its own "
            "side; this is a real statistical property, not a bug. Run "
            "alongside Local Moran's I rather than instead of it."
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
                self.FIELD,
                "Target numeric field to analyze",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial relationship / weights type",
                options=["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of neighbors (K value, KNN only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
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
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PERMUTATIONS,
                "Number of conditional permutations (per feature)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=199,
                minValue=99,
                maxValue=1999,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RANDOM_SEED,
                "Random seed (reproducibility)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=42,
                minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Cluster Analysis Output Layer",
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
                "Local Geary's C cluster and outlier classification report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_local_geary_c_report.html")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]

        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)

        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Target field '{field_name}' not found.")

        feedback.pushInfo("Generating spatial weights matrix...")
        neighbors, weights, id_order, _ = build_weights_matrix(
            source, weight_type, k_neighbors=k_neighbors, distance_band=distance_band, feedback=feedback
        )
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Extracting target field values...")
        y_dict = {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            val = f.attribute(field_name)
            if val is None or val == NULL or str(val) == "NULL":
                continue
            try:
                y_dict[f.id()] = float(val)
            except (ValueError, TypeError):
                continue

        valid_id_order = [fid for fid in id_order if fid in y_dict]
        y = np.array([y_dict[fid] for fid in valid_id_order])

        if len(y) <= 2:
            raise QgsProcessingException("At least 3 valid features with numeric values are required for Local Geary's C analysis.")

        feedback.pushInfo(f"Calculating Local Geary's C using {permutations} conditional permutations per feature...")
        local_c, z_scores, p_values, quadrants = calculate_local_geary_c(
            y, neighbors, weights, valid_id_order, permutations=permutations, seed=seed
        )

        class_summary = local_moran_class_summary(quadrants)
        feedback.pushInfo(class_summary["message"])

        if feedback.isCanceled():
            return {}

        out_fields = source.fields()
        out_fields.append(QgsField("lgc_c", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("lgc_z", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("lgc_p", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("quadrant", QVariant.String, len=20))
        out_fields.append(QgsField("lgc_nbrs", QVariant.Int))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs()
        )
        self.out_layer_id = dest_id

        results_map = {}
        valid_ids = set(valid_id_order)
        isolated_count = 0
        for idx, fid in enumerate(valid_id_order):
            neighbor_count = len([nid for nid in neighbors.get(fid, []) if nid in valid_ids])
            if neighbor_count == 0:
                isolated_count += 1
            results_map[fid] = (local_c[idx], z_scores[idx], p_values[idx], quadrants[idx], neighbor_count)
        if isolated_count:
            feedback.pushWarning(
                f"{isolated_count} feature(s) had no valid neighbors. Review lgc_nbrs and consider a larger distance band or K value."
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
                c_val, z, p, quad, neighbor_count = results_map[fid]
                out_feat.setAttribute("lgc_c", float(c_val))
                out_feat.setAttribute("lgc_z", float(z))
                out_feat.setAttribute("lgc_p", float(p))
                out_feat.setAttribute("quadrant", str(quad))
                out_feat.setAttribute("lgc_nbrs", int(neighbor_count))
            else:
                out_feat.setAttribute("lgc_c", None)
                out_feat.setAttribute("lgc_z", None)
                out_feat.setAttribute("lgc_p", None)
                out_feat.setAttribute("quadrant", "Not Significant")
                out_feat.setAttribute("lgc_nbrs", None)

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(50 + 50 * (current / total)))

        feedback.pushInfo("Generating HTML report...")
        self._write_html(html_path, field_name, len(y), weight_type, class_summary, isolated_count)

        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, field_name, n, weight_type, class_summary, isolated_count):
        counts = class_summary["counts"]
        donut_labels = [label for _, _, label in LISA_QUADRANT_STYLE]
        donut_values = [counts.get(code, 0) for code, _, _ in LISA_QUADRANT_STYLE]
        donut_colors = {label: color for code, color, label in LISA_QUADRANT_STYLE}

        kpi_row = kpi_card_row_html([
            {"label": "Cluster features", "value": str(class_summary["cluster_count"]), "sublabel": "HH + LL (similar to neighbors)", "tone": "good" if class_summary["cluster_count"] else "neutral"},
            {"label": "Outlier features", "value": str(class_summary["outlier_count"]), "sublabel": "HL + LH (dissimilar from neighbors)", "tone": "warn" if class_summary["outlier_count"] else "neutral"},
            {"label": "Significant total", "value": f"{class_summary['significant_count']} / {n}", "sublabel": f"Dominant: {class_summary['dominant_label']}"},
        ])

        guidance_html = analyst_guidance_html(
            "Local Geary's C",
            "Local Geary's C tests dissimilarity between each feature and its neighbors directly (via squared differences), then applies the same HH/LL/HL/LH sign classification Local Moran's I uses so results read the same way on the map.",
            [
                "lgc_nbrs is checked before trusting a 'Not Significant' result - zero valid neighbors looks identical to a genuine null result.",
                "A significant class forms a spatially coherent group with its neighbors, not an isolated single cell.",
                "The conditional-permutation count is high enough (199+ recommended) for a stable p-value.",
            ],
            [
                f"{isolated_count} feature(s) had zero valid neighbors" if isolated_count else "No isolated (zero-neighbor) features were found.",
                "Magnitude significance (Geary's C itself) and sign (HH/LL/HL/LH) are two separate tests that usually but not always agree.",
                "Each feature tested independently with no multiple-testing correction.",
            ],
            [
                "Local Moran's I as the standard cross-product-based comparison for the same clusters/outliers",
                "Getis-Ord Gi* for pure hot/cold magnitude ranking",
                "Bivariate LISA if the pattern should be tested against a second field",
            ],
            "Use a Local Geary's C cluster/outlier that agrees with Local Moran's I as stronger evidence than either statistic alone; a disagreement between the two is itself informative and worth a closer look.",
        )

        donut_html = donut_chart_svg(donut_labels, donut_values, colors=donut_colors, title="Local Geary's C Class Breakdown")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Local Geary's C Report</title>
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
        <h1>Local Geary's C (Cluster and Outlier Analysis)</h1>
        <p class="subtitle">Field Analyzed: <strong>{html.escape(field_name)}</strong> | Feature Count: <strong>{n}</strong> | Weights: <strong>{html.escape(weight_type)}</strong></p>
    </header>

    {kpi_row}

    <section>
        <h2>Local Geary's C Class Breakdown</h2>
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

        feedback.pushInfo("Applying Local Geary's C cluster analysis styling...")
        apply_output_metadata(
            layer,
            "PlanX GeoStats Local Geary's C cluster and outlier output",
            {
                "lgc_c": "Local Geary's C statistic (sum of neighbor squared differences)",
                "lgc_z": "Local Geary's C permutation z-score",
                "lgc_p": "Local Geary's C permutation p-value",
                "quadrant": "Classification: HH, LL, HL, LH, or Not Significant",
                "lgc_nbrs": "Valid neighbors used for the local statistic",
            },
            self.displayName(),
        )
        apply_renderer(layer, lisa_quadrant_renderer(layer.geometryType(), "quadrant"))

        return {}
