# -*- coding: utf-8 -*-
"""Network Reach Processing Algorithm."""
from __future__ import annotations

import logging

from ._mixins import HelpUrlMixin
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.network_engines import build_graph_from_lines, compute_reach
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class NetworkReachAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    RADIUS = "RADIUS"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "network_reach"

    def displayName(self) -> str:
        return "Network Reach"

    def group(self) -> str:
        return "07 | Network Centrality and Space Syntax"

    def groupId(self) -> str:
        return "planx_network_syntax"

    def icon(self):
        return algorithm_icon("network_reach")

    def createInstance(self):
        return NetworkReachAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Builds a graph from the input line layer and computes, for "
            "every node, how many other intersections/endpoints lie within "
            "Network distance radius of it along the network (not straight-"
            "line distance) - a simple, literal local-accessibility measure: "
            "'how much of the network can I reach within this walk/bike/"
            "drive distance from here'.\n\n"
            "Output: the input layer with reach_count (number of other nodes "
            "reachable within the radius; each segment's value is the "
            "average of its two endpoint nodes') and reach_max_dist (the "
            "network distance to the farthest node actually reached, useful "
            "to spot nodes near the edge of the study area whose true reach "
            "is undercounted because the network simply ends).\n\n"
            "Set Network distance radius in the same length units as your "
            "layer's CRS (meters for a typical projected CRS) - a common "
            "planning choice is 400-500m for a walkable distance or 800m for "
            "a longer walk/short bike trip. Low reach_count areas near the "
            "interior of the study area (not just near its edge) indicate "
            "genuinely poor local street connectivity, a useful input "
            "alongside Reach fields already present in some PlanX sample "
            "datasets (reach_500m_road_length, reach_800m_intersection) - "
            "this tool lets you recompute the same idea at any radius on "
            "your own network data."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input street/network line layer", [QgsProcessing.TypeVectorLine])
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RADIUS, "Network distance radius", type=QgsProcessingParameterNumber.Double,
                defaultValue=500.0, minValue=1.0,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output layer with network reach"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")
        radius = self.parameterAsDouble(parameters, self.RADIUS, context)

        feedback.pushInfo("Building network graph from line geometry (endpoint snapping)...")
        try:
            graph, edge_fid_map = build_graph_from_lines(source, feedback=feedback)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Network Reach", ["networkx"], exc))

        if graph.number_of_nodes() == 0:
            raise QgsProcessingException("No usable line segments were found (check that the layer has line geometry).")
        feedback.pushInfo(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

        feedback.pushInfo(f"Computing network reach within radius {radius}...")
        node_reach = compute_reach(graph, radius)
        count_values = {node: values["reach_count"] for node, values in node_reach.items()}
        dist_values = {node: values["reach_max_dist"] for node, values in node_reach.items()}

        from ..core.network_engines import edge_values_from_node_values
        edge_count = edge_values_from_node_values(edge_fid_map, count_values)
        edge_dist = edge_values_from_node_values(edge_fid_map, dist_values)

        out_fields = source.fields()
        out_fields.append(QgsField("reach_count", QVariant.Double, len=12, prec=2))
        out_fields.append(QgsField("reach_max_dist", QVariant.Double, len=12, prec=2))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        total = source.featureCount() or 1
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            out_feature.setAttribute("reach_count", edge_count.get(fid))
            out_feature.setAttribute("reach_max_dist", edge_dist.get(fid))
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats network reach output",
            {
                "reach_count": "Number of other nodes reachable within the network distance radius (averaged from this segment's two endpoints)",
                "reach_max_dist": "Network distance to the farthest node actually reached within the radius",
            },
            "network_reach",
        )
        return {}
