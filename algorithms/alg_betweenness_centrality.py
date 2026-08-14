# -*- coding: utf-8 -*-
"""Network Betweenness Centrality Processing Algorithm."""
from __future__ import annotations

from ._network_centrality_base import NetworkCentralityBase
from ._icons import algorithm_icon
from ..core.network_engines import compute_betweenness_centrality


class BetweennessCentralityAlgorithm(NetworkCentralityBase):
    FIELD_NAME = "betweenness"
    METRIC_LABEL = "Betweenness Centrality"

    def name(self) -> str:
        return "network_betweenness_centrality"

    def displayName(self) -> str:
        return "Network Betweenness Centrality"

    def group(self) -> str:
        return "07 | Network Centrality and Space Syntax"

    def groupId(self) -> str:
        return "planx_network_syntax"

    def icon(self):
        return algorithm_icon("network_betweenness_centrality")

    def createInstance(self):
        return BetweennessCentralityAlgorithm()

    def compute_node_values(self, graph):
        return compute_betweenness_centrality(graph)

    def shortHelpString(self) -> str:
        return (
            "Builds a graph from the input line layer (intersections and "
            "endpoints become nodes, segments become weighted edges - see "
            "Network Connectivity Diagnostics for details on how topology is "
            "recovered from the geometry) and computes each node's Freeman "
            "betweenness centrality: the fraction of all shortest paths "
            "between every other pair of nodes that pass through it. High "
            "betweenness marks segments that many routes are forced through - "
            "bottlenecks whose closure would disconnect or badly detour a "
            "large share of trips.\n\n"
            "This is classical shortest-path graph centrality, not angular "
            "segment ('choice') analysis from formal space syntax - it "
            "answers a related but distinct question using ordinary "
            "metric-distance shortest paths rather than turn-angle-weighted "
            "ones, and will not exactly match a NACH/choice value from "
            "Depthmap or a similar dedicated space-syntax tool.\n\n"
            "Output: the input layer with a betweenness field (each segment's "
            "value is the average of its two endpoint nodes' centrality). Use "
            "this to identify critical corridors for emergency-access "
            "planning, prioritizing street maintenance, or spotting where a "
            "closure would have outsized network impact.\n\n"
            "Runtime grows roughly with the square of the node count on dense "
            "networks; for a citywide network with many thousands of "
            "intersections, expect this to take noticeably longer than the "
            "other tools in this group."
        )
