# -*- coding: utf-8 -*-
"""Network Straightness Centrality Processing Algorithm."""
from __future__ import annotations

from ._network_centrality_base import NetworkCentralityBase
from ._icons import algorithm_icon
from ..core.network_engines import compute_straightness_centrality


class StraightnessCentralityAlgorithm(NetworkCentralityBase):
    FIELD_NAME = "straightness"
    METRIC_LABEL = "Straightness Centrality"

    def name(self) -> str:
        return "network_straightness_centrality"

    def displayName(self) -> str:
        return "Network Straightness Centrality"

    def group(self) -> str:
        return "07 | Network Centrality and Space Syntax"

    def groupId(self) -> str:
        return "planx_network_syntax"

    def icon(self):
        return algorithm_icon("network_straightness_centrality")

    def createInstance(self):
        return StraightnessCentralityAlgorithm()

    def compute_node_values(self, graph):
        return compute_straightness_centrality(graph)

    def shortHelpString(self) -> str:
        return (
            "Builds a graph from the input line layer and computes each "
            "node's straightness centrality (Vragovic et al. 2005): for "
            "every other reachable node, the ratio of straight-line "
            "(Euclidean) distance to actual shortest-path network distance, "
            "averaged across all of them. A value near 1 means routes from "
            "this location are nearly as direct as flying in a straight "
            "line; a low value means reaching other places typically "
            "requires significant detour relative to the direct distance.\n\n"
            "This is the closest of the tools in this group to the "
            "'integration' intuition from space syntax - both ask how "
            "efficiently a location connects to the rest of the study area - "
            "but it is computed from ordinary metric shortest paths, not the "
            "turn-angle-weighted axial/segment analysis a dedicated "
            "space-syntax tool (Depthmap, depthmapX, or the swintX/COSMic "
            "family) uses; do not present straightness values as equivalent "
            "to NACH or a formal integration score.\n\n"
            "Output: the input layer with a straightness field (each "
            "segment's value is the average of its two endpoint nodes'). Low "
            "straightness areas are where the street network forces "
            "meaningful detours - useful for spotting where a new "
            "connection would most improve directness, or explaining why "
            "walk/bike trips in an area run longer than straight-line "
            "distance would suggest.\n\n"
            "Like Betweenness Centrality, this computes all-pairs shortest "
            "paths and will be noticeably slower on networks with many "
            "thousands of nodes."
        )
