# -*- coding: utf-8 -*-
"""Shared processAlgorithm implementation for the single-value node-centrality
tools (Betweenness / Closeness / Straightness). Each concrete alg_*.py file
is a thin subclass that sets METRIC_FN/FIELD_NAME and implements the
QGIS-required name()/displayName()/group()/icon()/createInstance() methods
directly, plus its own shortHelpString() (the provider-catalog smoke test
parses each algorithm file's own class body for those)."""
from __future__ import annotations

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
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.network_engines import build_graph_from_lines, edge_values_from_node_values
from ..dependencies import optional_dependency_error


class NetworkCentralityBase(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    FIELD_NAME = "centrality"
    METRIC_LABEL = "Centrality"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input street/network line layer", [QgsProcessing.TypeVectorLine])
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, f"Output layer with {self.FIELD_NAME}"))

    def compute_node_values(self, graph):
        raise NotImplementedError

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        feedback.pushInfo("Building network graph from line geometry (endpoint snapping)...")
        try:
            graph, edge_fid_map = build_graph_from_lines(source, feedback=feedback)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error(self.displayName(), ["networkx"], exc))

        if graph.number_of_nodes() == 0:
            raise QgsProcessingException("No usable line segments were found (check that the layer has line geometry).")
        feedback.pushInfo(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

        feedback.pushInfo(f"Computing {self.METRIC_LABEL}...")
        node_values = self.compute_node_values(graph)
        edge_values = edge_values_from_node_values(edge_fid_map, node_values)

        out_fields = source.fields()
        out_fields.append(QgsField(self.FIELD_NAME, QVariant.Double, len=14, prec=8))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        total = source.featureCount() or 1
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            out_feature.setAttribute(self.FIELD_NAME, edge_values.get(feature.id()))
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            f"PlanX GeoStats {self.METRIC_LABEL} output",
            {self.FIELD_NAME: f"{self.METRIC_LABEL}, averaged from this segment's two endpoint nodes"},
            self.FIELD_NAME,
        )
        return {}
