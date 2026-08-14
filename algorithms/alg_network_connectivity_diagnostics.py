# -*- coding: utf-8 -*-
"""Network Connectivity Diagnostics Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

from ._mixins import HelpUrlMixin
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFileDestination,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.network_engines import build_graph_from_lines, network_connectivity_summary
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class NetworkConnectivityDiagnosticsAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "network_connectivity_diagnostics"

    def displayName(self) -> str:
        return "Network Connectivity Diagnostics"

    def group(self) -> str:
        return "07 | Network Centrality and Space Syntax"

    def groupId(self) -> str:
        return "planx_network_syntax"

    def icon(self):
        return algorithm_icon("network_connectivity_diagnostics")

    def createInstance(self):
        return NetworkConnectivityDiagnosticsAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Run this first, before any other tool in this group. It builds "
            "the same graph the centrality and reach tools use (line "
            "endpoints within 3 decimal digits of each other in the layer's "
            "CRS units are snapped together into one intersection node - a "
            "street layer with small digitizing gaps at intersections will "
            "silently produce disconnected fragments here rather than a "
            "connected network) and reports whether that graph is actually "
            "one connected network or several disconnected pieces, plus dead-"
            "end and node-degree statistics.\n\n"
            "Output: the input layer with a seg_component_id field (segments "
            "sharing a component id are part of the same connected sub-"
            "network - map this categorically to see disconnected fragments "
            "at a glance), plus an HTML report with node/edge counts, "
            "component count and sizes, dead-end count, and node-degree "
            "summary statistics.\n\n"
            "Betweenness, Closeness, and Straightness Centrality all compute "
            "distances within each node's own connected component only - if "
            "this report shows more than one component, values from "
            "different components are not on a comparable scale, and a small "
            "disconnected fragment can produce misleadingly extreme "
            "centrality values within its own tiny component. If the largest "
            "component is much smaller than the total node count, check the "
            "source line layer for digitizing gaps at intersections before "
            "trusting downstream centrality results."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input street/network line layer", [QgsProcessing.TypeVectorLine])
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output layer with connected component id"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output connectivity diagnostics HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Network connectivity diagnostics report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_network_connectivity_report.html")

        feedback.pushInfo("Building network graph from line geometry (endpoint snapping)...")
        try:
            graph, edge_fid_map = build_graph_from_lines(source, feedback=feedback)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Network Connectivity Diagnostics", ["networkx"], exc))

        if graph.number_of_nodes() == 0:
            raise QgsProcessingException("No usable line segments were found (check that the layer has line geometry).")

        summary = network_connectivity_summary(graph)
        feedback.pushInfo(
            f"{summary['n_nodes']} nodes, {summary['n_edges']} edges, {summary['n_components']} connected component(s); "
            f"largest = {summary['largest_component_size']} nodes; {summary['dead_end_count']} dead end(s)."
        )
        if summary["n_components"] > 1:
            feedback.pushWarning(
                f"Network is split into {summary['n_components']} disconnected components - "
                "centrality values from different components are not comparable."
            )

        node_component = summary["node_component"]
        edge_component = {}
        for fid, pairs in edge_fid_map.items():
            for u, _ in pairs:
                edge_component[fid] = node_component.get(u)
                break

        out_fields = source.fields()
        out_fields.append(QgsField("seg_component_id", QVariant.Int))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        total = source.featureCount() or 1
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            out_feature.setAttribute("seg_component_id", edge_component.get(feature.id()))
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, summary)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats network connectivity diagnostics output",
            {"seg_component_id": "Connected-component index; segments sharing a value belong to the same connected sub-network"},
            "network_connectivity_diagnostics",
        )
        return {}

    def _write_html(self, path, summary):
        component_rows = "".join(
            f"<tr><td>{i}</td><td>{size}</td></tr>" for i, size in enumerate(summary["component_sizes"][:20])
        )
        guidance = analyst_guidance_html(
            "Network Connectivity Diagnostics",
            "Reports whether the recovered graph is one connected network or "
            "several fragments, and basic node-degree/dead-end statistics.",
            [
                "n_components = 1 (or the non-largest components are intentionally isolated features, like a disconnected cul-de-sac loop).",
                "Dead-end count matches what you would expect from the real street network.",
            ],
            [
                "n_components > 1 with several similarly-sized components (likely digitizing gaps at intersections, not real disconnection).",
                "Largest component size is much smaller than total node count.",
            ],
            [
                "Snap/fix intersection gaps in the source line layer, then rerun this diagnostic before trusting centrality tools.",
                "Betweenness / Closeness / Straightness Centrality - only meaningful once connectivity looks right.",
            ],
            "Do not run the centrality tools on a network with unexpected "
            "multiple components without first checking whether that "
            "reflects real disconnection or a digitizing gap.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Network Connectivity Diagnostics</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 8px 10px; text-align: left; font-size: .9rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
</style></head>
<body><div class="container">
<h1>Network Connectivity Diagnostics</h1>
<div class="summary">
Nodes = {summary['n_nodes']} | Edges = {summary['n_edges']} | Components = {summary['n_components']}<br>
Largest component = {summary['largest_component_size']} nodes | Dead ends = {summary['dead_end_count']}<br>
Node degree: min={summary['degree_min']}, median={summary['degree_median']:.1f}, max={summary['degree_max']}
</div>
<h2>Component Sizes (largest 20)</h2>
<table><thead><tr><th>Component</th><th>Node count</th></tr></thead><tbody>{component_rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
