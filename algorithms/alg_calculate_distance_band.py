# -*- coding: utf-8 -*-
"""Calculate Distance Band from Neighbor Count Processing Algorithm."""
from __future__ import annotations

import logging
import os
import tempfile
import numpy as np

from ._mixins import HelpUrlMixin
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFile,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputHtml,
    QgsWkbTypes,
)

from ..core.stats_engines import calculate_distance_band_stats
from ..core.weights import (
    build_weights_matrix,
    export_weights_csv,
    geometry_centroid_point,
    import_weights_csv,
    weights_diagnostics,
)

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class CalculateDistanceBandAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    NEIGHBOR_COUNT = "NEIGHBOR_COUNT"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    DISTANCE_BAND = "DISTANCE_BAND"
    NEIGHBOR_ORDER = "NEIGHBOR_ORDER"
    REPAIR_ISLANDS = "REPAIR_ISLANDS"
    DISTANCE_DECAY = "DISTANCE_DECAY"
    IMPORT_CSV = "IMPORT_CSV"
    EXPORT_CSV = "EXPORT_CSV"
    OUTPUT_NODES = "OUTPUT_NODES"
    OUTPUT_EDGES = "OUTPUT_EDGES"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "calculate_distance_band"

    def displayName(self) -> str:
        return "Spatial Weights Workbench and Distance Band"

    def group(self) -> str:
        return "01 | Data Preparation and Neighborhoods"

    def groupId(self) -> str:
        return "planx_prepare_neighbors"

    def icon(self):
        return algorithm_icon("calculate_distance_band")

    def createInstance(self):
        return CalculateDistanceBandAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes the distance to the k-th nearest neighbor for every feature "
            "(centroid-to-centroid, Euclidean) and reports six summary statistics: "
            "minimum, 25th percentile, median, mean, 75th percentile, and maximum. "
            "The maximum is the smallest distance band that guarantees every feature "
            "has at least k neighbors - the data-driven answer to 'what distance "
            "band should I use?'.\n\n"
            "k = 1 answers the most common question, 'what distance ensures every "
            "feature has at least one neighbor?'. k = 5 to 8 produces a neighborhood "
            "size comparable to Queen contiguity on a typical polygon layer. Larger k "
            "smooths local variation aggressively and should be justified by the "
            "scale of the spatial process being studied.\n\n"
            "If the mean is more than twice the median, the feature distribution is "
            "strongly heterogeneous - a dense core with distant outliers - and a "
            "single distance band will give core features hundreds of neighbors "
            "while peripheral features get exactly k. Consider KNN weights or "
            "splitting the analysis into sub-regions instead.\n\n"
            "This tool guarantees connectivity, not that the spatial process peaks "
            "at that scale. Follow up with Incremental Spatial Autocorrelation on "
            "the actual target variable to find the analytically optimal band; when "
            "the two differ substantially, prefer the autocorrelation-peak distance "
            "and note the isolation count it leaves behind. The workbench also builds "
            "Queen, Rook, KNN, or distance-band graphs; supports higher-order neighbors, "
            "inverse-distance decay, nearest-neighbor island repair, portable CSV "
            "import/export, node diagnostics, connected components, and a visual edge layer."
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
            QgsProcessingParameterNumber(
                self.NEIGHBOR_COUNT,
                "Neighbor count (N)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1
            )
        )
        self.addParameter(QgsProcessingParameterEnum(
            self.WEIGHT_TYPE, "Weights graph to inspect",
            options=["Queen contiguity", "Rook contiguity", "KNN", "Distance band"],
            defaultValue=2,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.DISTANCE_BAND, "Distance threshold (0 = use recommended maximum)",
            type=QgsProcessingParameterNumber.Double, defaultValue=0.0, minValue=0.0,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.NEIGHBOR_ORDER, "Neighbor order", type=QgsProcessingParameterNumber.Integer,
            defaultValue=1, minValue=1, maxValue=5,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.REPAIR_ISLANDS, "Connect islands to nearest feature", defaultValue=False,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.DISTANCE_DECAY, "Inverse-distance decay power (0 = binary)",
            type=QgsProcessingParameterNumber.Double, defaultValue=0.0, minValue=0.0, maxValue=4.0,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.IMPORT_CSV, "Import weights CSV instead of building",
            behavior=QgsProcessingParameterFile.File, extension="csv", optional=True,
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.EXPORT_CSV, "Export weights adjacency CSV", fileFilter="CSV files (*.csv)", optional=True,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_NODES, "Weights node diagnostics", optional=True,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_EDGES, "Weights graph edges", optional=True,
        ))
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
                "Distance band statistics report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        k_neighbors = self.parameterAsInt(parameters, self.NEIGHBOR_COUNT, context)

        # Resolve output HTML file path
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, "distance_band_report.html")

        # Extract centroids
        x_coords, y_coords, feature_ids = [], [], []

        feedback.pushInfo("Extracting feature centroids...")
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom = f.geometry()
            if geom is None or geom.isEmpty():
                continue

            centroid = geometry_centroid_point(geom)
            if centroid is None:
                continue
            x_coords.append(centroid.x())
            y_coords.append(centroid.y())
            feature_ids.append(f.id())
            feedback.setProgress(int(40 * (idx / total)))

        n_feats = len(x_coords)
        if n_feats <= k_neighbors:
            raise QgsProcessingException(
                f"Feature count ({n_feats}) must be greater than neighbor count ({k_neighbors}) to compute stats."
            )

        x_arr = np.array(x_coords)
        y_arr = np.array(y_coords)

        feedback.pushInfo("Computing pairwise distance bands...")
        stats = calculate_distance_band_stats(x_arr, y_arr, k_neighbors)

        if feedback.isCanceled():
            return {}

        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)]
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context) or stats["max"]
        neighbor_order = self.parameterAsInt(parameters, self.NEIGHBOR_ORDER, context)
        repair_islands = self.parameterAsBoolean(parameters, self.REPAIR_ISLANDS, context)
        decay_power = self.parameterAsDouble(parameters, self.DISTANCE_DECAY, context)
        import_path = self.parameterAsFile(parameters, self.IMPORT_CSV, context)

        if import_path:
            neighbors, weights = import_weights_csv(import_path, feature_ids)
            id_order = feature_ids
            graph_source = "imported CSV"
        else:
            neighbors, weights, id_order, _ = build_weights_matrix(
                source,
                weight_type,
                k_neighbors=k_neighbors,
                distance_band=distance_band,
                neighbor_order=neighbor_order,
                repair_islands=repair_islands,
                distance_decay_power=decay_power,
                feedback=feedback,
            )
            graph_source = weight_type

        diagnostics = weights_diagnostics(neighbors)
        export_path = self.parameterAsFileOutput(parameters, self.EXPORT_CSV, context)
        if export_path:
            export_weights_csv(export_path, neighbors, weights)

        node_fields = source.fields()
        node_fields.append(QgsField("w_nbrs", QVariant.Int))
        node_fields.append(QgsField("w_island", QVariant.Int))
        node_fields.append(QgsField("w_order", QVariant.Int))
        node_sink, node_dest = self.parameterAsSink(
            parameters, self.OUTPUT_NODES, context, node_fields,
            source.wkbType(), source.sourceCrs(),
        )
        if node_sink is not None:
            for feature in source.getFeatures():
                output = QgsFeature(feature)
                output.setFields(node_fields)
                count = len(neighbors.get(feature.id(), []))
                output.setAttribute("w_nbrs", count)
                output.setAttribute("w_island", int(count == 0))
                output.setAttribute("w_order", neighbor_order)
                node_sink.addFeature(output, QgsFeatureSink.FastInsert)

        edge_fields = QgsFields()
        edge_fields.append(QgsField("source_id", QVariant.LongLong))
        edge_fields.append(QgsField("target_id", QVariant.LongLong))
        edge_fields.append(QgsField("weight", QVariant.Double))
        edge_sink, edge_dest = self.parameterAsSink(
            parameters, self.OUTPUT_EDGES, context, edge_fields,
            QgsWkbTypes.LineString, source.sourceCrs(),
        )
        centroid_map = {
            feature.id(): geometry_centroid_point(feature.geometry())
            for feature in source.getFeatures()
        }
        if edge_sink is not None:
            for source_id in id_order:
                start = centroid_map.get(source_id)
                if start is None:
                    continue
                for target_id, weight in zip(neighbors.get(source_id, []), weights.get(source_id, [])):
                    end = centroid_map.get(target_id)
                    if end is None:
                        continue
                    edge = QgsFeature(edge_fields)
                    edge.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(start), QgsPointXY(end)]))
                    edge.setAttributes([source_id, target_id, float(weight)])
                    edge_sink.addFeature(edge, QgsFeatureSink.FastInsert)

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, n_feats, k_neighbors, stats, diagnostics, graph_source,
            neighbor_order, decay_power, repair_islands,
        )

        return {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
            self.EXPORT_CSV: export_path,
            self.OUTPUT_NODES: node_dest,
            self.OUTPUT_EDGES: edge_dest,
        }

    def write_html_report(self, path, n, k, stats, diagnostics, graph_source, order, decay, repaired):
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Distance Band Report</title>
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
        max-width: 760px;
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
        font-size: 1.6rem;
    }}
    .subtitle {{
        color: #718096;
        margin: 0;
        font-size: 0.95rem;
    }}
    .advice-box {{
        background-color: #ebf8ff;
        border-left: 5px solid #3182ce;
        padding: 15px 20px;
        border-radius: 4px;
        margin-bottom: 30px;
        font-size: 0.9rem;
        color: #2b6cb0;
    }}
    .advice-box strong {{
        color: #2c5282;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 25px;
    }}
    th, td {{
        padding: 12px 15px;
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
    .metric-name {{
        font-weight: 600;
        color: #2d3748;
    }}
    .metric-val {{
        font-family: monospace;
        font-size: 1rem;
        font-weight: 600;
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
        <h1>Distance Band Statistics Summary</h1>
        <p class="subtitle">Feature count: <strong>{n}</strong> | Neighbor count: <strong>{k}</strong></p>
    </header>

    <div class="advice-box">
        <strong>Recommendation:</strong> To ensure that every feature in the dataset has at least <strong>{k}</strong> neighbor(s) during analysis, select a threshold distance band of at least <strong>{stats["max"]:.6f}</strong> map units.
    </div>

    <h2>Weights Graph Diagnostics</h2>
    <table><tbody>
      <tr><td class="metric-name">Graph source</td><td>{graph_source}</td></tr>
      <tr><td class="metric-name">Neighbor order / decay</td><td>{order} / {decay:.3g}</td></tr>
      <tr><td class="metric-name">Island repair</td><td>{'Enabled' if repaired else 'Disabled'}</td></tr>
      <tr><td class="metric-name">Directed links</td><td>{diagnostics['edge_count']}</td></tr>
      <tr><td class="metric-name">Islands</td><td>{diagnostics['island_count']}</td></tr>
      <tr><td class="metric-name">Connected components</td><td>{diagnostics['component_count']}</td></tr>
      <tr><td class="metric-name">Neighbor count min / mean / max</td><td>{diagnostics['min_neighbors']} / {diagnostics['mean_neighbors']:.2f} / {diagnostics['max_neighbors']}</td></tr>
      <tr><td class="metric-name">Asymmetric directed links</td><td>{diagnostics['asymmetric_links']}</td></tr>
    </tbody></table>

    <table>
        <thead>
            <tr>
                <th>Neighbor Distance Parameter</th>
                <th>Distance (Map Units)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="metric-name">Minimum Distance to {k}-th Neighbor</td>
                <td class="metric-val">{stats["min"]:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">Average Distance to {k}-th Neighbor</td>
                <td class="metric-val">{stats["mean"]:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">25th Percentile (p25)</td>
                <td class="metric-val">{stats["p25"]:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">50th Percentile / Median Distance</td>
                <td class="metric-val">{stats["median"]:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">75th Percentile (p75)</td>
                <td class="metric-val">{stats["p75"]:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">Maximum Distance to {k}-th Neighbor (Ensures Connectivity)</td>
                <td class="metric-val" style="color: #c53030;">{stats["max"]:.6f}</td>
            </tr>
        </tbody>
    </table>

    <footer>
        Generated by PlanX GeoStats Lab spatial component utilities.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
