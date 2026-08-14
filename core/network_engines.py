# -*- coding: utf-8 -*-
"""Graph construction and centrality helpers for the PlanX GeoStats Lab
Network Centrality and Space Syntax group.

These compute classical graph-theory (Freeman) centrality on the primal
street-segment graph (nodes = intersections/endpoints, edges = segments
weighted by length) via networkx. This is NOT angular segment / axial-line
space syntax analysis (the kind that produces integration/choice values
comparable to Depthmap or a NACH/NAIN field already baked into a dataset) -
it answers a related but distinct question using standard shortest-path
graph metrics, and tools built on it say so explicitly.

networkx is imported lazily so this module (and the Processing provider)
stays importable when the optional package is missing.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

Node = Tuple[float, float]


def _part_length(part) -> float:
    total = 0.0
    for i in range(len(part) - 1):
        total += part[i].distance(part[i + 1])
    return total


def build_graph_from_lines(source, snap_precision: int = 3, feedback=None):
    """Build an undirected weighted networkx graph from a line layer.

    Endpoints within snap_precision decimal digits of each other are treated
    as the same node (coordinate snapping - the standard way to recover
    topology from unstructured line geometry). Returns (graph, edge_fid_map)
    where edge_fid_map maps each source feature id to the list of (u, v)
    node-pairs it contributed (a multipart line contributes one pair per part).
    """
    import networkx as nx

    graph = nx.Graph()
    edge_fid_map: Dict[int, list] = {}
    total = source.featureCount() or 1

    for idx, feature in enumerate(source.getFeatures()):
        if feedback and feedback.isCanceled():
            break
        geom = feature.geometry()
        if geom is None or geom.isEmpty():
            continue
        parts = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
        for part in parts:
            if len(part) < 2:
                continue
            start: Node = (round(part[0].x(), snap_precision), round(part[0].y(), snap_precision))
            end: Node = (round(part[-1].x(), snap_precision), round(part[-1].y(), snap_precision))
            if start == end:
                continue
            length = _part_length(part)
            if length <= 0:
                continue
            if graph.has_edge(start, end):
                if length < graph[start][end]["weight"]:
                    graph[start][end]["weight"] = length
            else:
                graph.add_edge(start, end, weight=length)
            edge_fid_map.setdefault(feature.id(), []).append((start, end))
        if feedback:
            feedback.setProgress(int(30 * (idx / total)))

    return graph, edge_fid_map


def edge_values_from_node_values(edge_fid_map: Dict[int, list], node_values: Dict[Node, float]) -> Dict[int, float]:
    """Average the two endpoint node values for every edge belonging to each
    source feature id (a feature can contribute more than one part)."""
    result = {}
    for fid, pairs in edge_fid_map.items():
        values = []
        for u, v in pairs:
            uv = [node_values.get(u), node_values.get(v)]
            uv = [value for value in uv if value is not None]
            if uv:
                values.append(float(np.mean(uv)))
        if values:
            result[fid] = float(np.mean(values))
    return result


def compute_betweenness_centrality(graph) -> Dict[Node, float]:
    import networkx as nx
    return nx.betweenness_centrality(graph, weight="weight", normalized=True)


def compute_closeness_centrality(graph) -> Dict[Node, float]:
    import networkx as nx
    return nx.closeness_centrality(graph, distance="weight")


def compute_straightness_centrality(graph) -> Dict[Node, float]:
    """Straightness centrality (Vragovic et al. 2005): for each node, the
    average ratio of Euclidean distance to shortest network-path distance to
    every other reachable node - close to 1 means routes to other places are
    nearly as direct as a straight line (a proxy for the 'integration' idea
    in space syntax, computed via ordinary shortest paths rather than angular
    segment analysis)."""
    import networkx as nx

    result = {}
    for node in graph.nodes():
        lengths = nx.single_source_dijkstra_path_length(graph, node, weight="weight")
        total, count = 0.0, 0
        for other, net_dist in lengths.items():
            if other == node or net_dist <= 0:
                continue
            euclidean = ((node[0] - other[0]) ** 2 + (node[1] - other[1]) ** 2) ** 0.5
            total += euclidean / net_dist
            count += 1
        result[node] = (total / count) if count else 0.0
    return result


def compute_reach(graph, radius: float) -> Dict[Node, dict]:
    """Reach: how many other nodes (and how much network length) lie within
    a given network-distance radius of each node - a simple, interpretable
    local-accessibility measure."""
    import networkx as nx

    result = {}
    for node in graph.nodes():
        lengths = nx.single_source_dijkstra_path_length(graph, node, cutoff=radius, weight="weight")
        reachable = {other: dist for other, dist in lengths.items() if other != node}
        result[node] = {"reach_count": len(reachable), "reach_max_dist": max(reachable.values()) if reachable else 0.0}
    return result


def network_connectivity_summary(graph) -> dict:
    import networkx as nx

    components = list(nx.connected_components(graph))
    component_sizes = sorted((len(component) for component in components), reverse=True)
    degrees = [degree for _, degree in graph.degree()]
    dead_ends = sum(1 for degree in degrees if degree == 1)
    node_component = {}
    for component_id, component in enumerate(components):
        for node in component:
            node_component[node] = component_id
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_components": len(components),
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "component_sizes": component_sizes,
        "dead_end_count": dead_ends,
        "degree_min": int(np.min(degrees)) if degrees else 0,
        "degree_median": float(np.median(degrees)) if degrees else 0.0,
        "degree_max": int(np.max(degrees)) if degrees else 0,
        "node_component": node_component,
    }
