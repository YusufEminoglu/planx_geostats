# -*- coding: utf-8 -*-
"""Network Closeness Centrality Processing Algorithm."""
from __future__ import annotations

from ._network_centrality_base import NetworkCentralityBase
from ._icons import algorithm_icon
from ..core.network_engines import compute_closeness_centrality


class ClosenessCentralityAlgorithm(NetworkCentralityBase):
    FIELD_NAME = "closeness"
    METRIC_LABEL = "Closeness Centrality"

    def name(self) -> str:
        return "network_closeness_centrality"

    def displayName(self) -> str:
        return "Network Closeness Centrality"

    def group(self) -> str:
        return "07 | Network Centrality and Space Syntax"

    def groupId(self) -> str:
        return "planx_network_syntax"

    def icon(self):
        return algorithm_icon("network_closeness_centrality")

    def createInstance(self):
        return ClosenessCentralityAlgorithm()

    def compute_node_values(self, graph):
        return compute_closeness_centrality(graph)

    def shortHelpString(self) -> str:
        return (
            "Builds a graph from the input line layer and computes each "
            "node's closeness centrality: the inverse of the average "
            "shortest-path network distance from that node to every other "
            "reachable node. A high value means the location is, on average, "
            "network-close to everywhere else in the connected part of the "
            "study area - a candidate location for a facility meant to "
            "minimize average travel distance across the whole network.\n\n"
            "Where Betweenness Centrality answers 'how much through-traffic "
            "is forced past this point', Closeness answers 'how convenient "
            "is this point as an average destination or origin' - the two "
            "frequently disagree (a well-connected suburban node can be "
            "close to everything without carrying much through-traffic, and "
            "a bridge segment can carry heavy through-traffic without being "
            "particularly close to anywhere).\n\n"
            "Output: the input layer with a closeness field (each segment's "
            "value is the average of its two endpoint nodes'). If the network "
            "has multiple disconnected components (see Network Connectivity "
            "Diagnostics), closeness is computed only within each node's own "
            "component - values are not comparable across components of very "
            "different size."
        )
