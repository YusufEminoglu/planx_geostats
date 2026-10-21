# -*- coding: utf-8 -*-
"""REDCAP regionalization Processing algorithm."""
from __future__ import annotations

import html

from ..core.charts import kpi_card_row_html
from ..core.regionalization_engines import redcap
from ..core.reporting import analyst_guidance_html
from ._local_common import DOUBLE, INT, SpatialStatBase, ToolResult, callout, fmt, report_page
from ._region_common import REGION_DOCS, quality_section, region_compactness

LINKAGES = [
    ("Full-order complete linkage (Guo's best performer)", "complete"),
    ("Full-order average linkage", "average"),
    ("Full-order Ward linkage", "ward"),
    ("Full-order single linkage", "single"),
]


class RedcapAlgorithm(SpatialStatBase):
    FIELDS = "FIELDS"
    K_REGIONS = "K_REGIONS"
    LINKAGE = "LINKAGE"
    MIN_SIZE = "MIN_SIZE"

    ALG_ID = "redcap_regionalization"
    DISPLAY = "REDCAP Regionalization (Contiguity-Constrained Clustering)"
    GROUP = "03 | Hot Spots and Spatial Outliers"
    GROUP_ID = "planx_hotspots_outliers"
    OUTPUT_LABEL = "REDCAP regions"
    REPORT_LABEL = "REDCAP report"
    HELP = (
        "REDCAP (Guo 2008): regionalisation with a contiguity-constrained hierarchical clustering. Stage 1 "
        "merges neighbouring clusters bottom-up - only clusters that touch may merge - using a linkage computed "
        "over every pair of units in the two clusters (full order), and records the edge that joined them, "
        "building a spanning tree. Stage 2 cuts that tree into k regions, each time removing the edge whose "
        "removal most reduces the within-region sum of squares.\n\n"
        "SKATER builds its tree from a minimum spanning tree of the contiguity graph and so only ever looks at "
        "neighbouring pairs; REDCAP's full-order linkage looks at whole clusters, which Guo found gives more "
        "homogeneous regions, with full-order complete linkage the best of the variants. The method is "
        "deterministic - no seed, the same input always gives the same regions - and keeps an n x n distance "
        "matrix, so it is limited to 4000 units.\n\n"
        "Output: region_id and silhouette scores; the report has the full partition-quality assessment."
    )

    def add_parameters(self):
        self.numeric_field(self.FIELDS, "Attribute fields regions should be homogeneous on", multiple=True)
        self.integer(self.K_REGIONS, "Number of regions (k)", 5, minimum=1)
        self.enum(self.LINKAGE, "Linkage", [label for label, _ in LINKAGES], 0)

    def add_trailing_parameters(self):
        self.integer(self.MIN_SIZE, "Minimum units per region", 1, minimum=1)

    def compute(self, source, parameters, context, feedback):
        names = self.field_list(parameters, self.FIELDS, context)
        self.require(names, "Select at least one attribute field.")
        k = self.parameterAsInt(parameters, self.K_REGIONS, context)
        linkage_label, linkage = LINKAGES[self.parameterAsEnum(parameters, self.LINKAGE, context)]
        min_size = self.parameterAsInt(parameters, self.MIN_SIZE, context)
        fids, x, dropped = self.read_matrix(source, names, feedback)
        self.require(len(fids) >= k, f"Only {len(fids)} complete features for k = {k} regions.")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo(f"REDCAP: building the {linkage} spanning tree and cutting it into {k} regions...")
        try:
            res = redcap(x, graph, k, linkage=linkage, min_size=min_size)
        except ValueError as exc:
            self.require(False, str(exc))
        labels = res["labels"]
        section, per_unit, _summary = quality_section(x, labels, graph, compactness=region_compactness(source, fids, labels))
        rows = {fid: {"region_id": int(labels[i]), "sil": per_unit["sil"][i], "bnd_sil": per_unit["bnd_sil"][i],
                      "path_sil": per_unit["path_sil"][i]} for i, fid in enumerate(fids)}
        kpis = kpi_card_row_html([
            {"label": "Regions", "value": str(res["k"])},
            {"label": "Within-region SSD", "value": fmt(res["ssd"])},
            {"label": "Linkage", "value": linkage},
        ])
        guidance = analyst_guidance_html(
            "REDCAP",
            "REDCAP regions are contiguous and are cut from a tree that joined the most similar neighbouring clusters "
            "first, so each region is a coherent branch of that tree.",
            ["Mean silhouette is positive.", "The linkage choice does not change the main boundaries.",
             "k matches the planning question."],
            ["Single linkage chains: long thin regions are a sign to switch to complete or Ward.",
             "A tree cut is greedy - a boundary set early is never revisited (AZP can refine it).",
             "Above 4000 units the distance matrix is too large; use Max-p or AZP."],
            ["SKATER Regionalization for the minimum-spanning-tree counterpart", "AZP to refine REDCAP's boundaries",
             "Regionalization Quality Assessment to compare partitions"],
            "Use REDCAP when a reproducible, explainable zoning is needed - the same data always gives the same regions.",
        )
        subtitle = (f"Fields: <strong>{html.escape(', '.join(names))}</strong> | k = <strong>{k}</strong> | Linkage: "
                    f"<strong>{html.escape(linkage_label)}</strong> | Weights: <strong>{html.escape(weight_label)}</strong>")
        drop = callout(f"{dropped} feature(s) with missing values were left out.") if dropped else ""
        return ToolResult(report_html=report_page(self.DISPLAY, subtitle, kpis + drop + section + guidance),
                          fields=[("region_id", INT), ("sil", DOUBLE), ("bnd_sil", DOUBLE), ("path_sil", DOUBLE)],
                          rows=rows, field_docs=dict(REGION_DOCS), style=("cluster_id", "region_id"),
                          layer_title="PlanX GeoStats REDCAP regions")
