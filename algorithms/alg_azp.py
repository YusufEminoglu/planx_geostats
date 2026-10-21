# -*- coding: utf-8 -*-
"""Automatic Zoning Procedure (AZP) Processing algorithm."""
from __future__ import annotations

import html

from ..core.charts import kpi_card_row_html
from ..core.regionalization_engines import azp
from ..core.reporting import analyst_guidance_html
from ._local_common import DOUBLE, INT, SpatialStatBase, ToolResult, callout, fmt, report_page
from ._region_common import REGION_DOCS, quality_section, region_compactness

METHODS = [
    ("Basic AZP (improving moves only)", "greedy"),
    ("AZP with simulated annealing", "sa"),
    ("AZP with tabu search", "tabu"),
]


class AZPAlgorithm(SpatialStatBase):
    FIELDS = "FIELDS"
    K_REGIONS = "K_REGIONS"
    METHOD = "METHOD"
    RESTARTS = "RESTARTS"
    MIN_SIZE = "MIN_SIZE"
    RANDOM_SEED = "RANDOM_SEED"

    ALG_ID = "azp_regionalization"
    DISPLAY = "AZP Regionalization (Automatic Zoning Procedure)"
    GROUP = "03 | Hot Spots and Spatial Outliers"
    GROUP_ID = "planx_hotspots_outliers"
    OUTPUT_LABEL = "AZP regions"
    REPORT_LABEL = "AZP report"
    HELP = (
        "Openshaw's Automatic Zoning Procedure (1977; Openshaw & Rao 1995): aggregate the units into exactly "
        "k contiguous regions that are as homogeneous as possible on the attribute fields. AZP starts from a "
        "random contiguous partition and keeps moving units across region borders whenever the move lowers "
        "the total within-region sum of squares and leaves the donor region connected. Because the result "
        "depends on the start, the tool runs several random starts and keeps the best.\n\n"
        "Variants: basic AZP accepts only improving moves and stops at the first local optimum; simulated "
        "annealing sometimes accepts a worsening move early on to escape it; tabu search always makes the "
        "best admissible move and forbids undoing recent moves. Use AZP when the number of regions is fixed "
        "in advance (e.g. k planning districts); use Max-p when a minimum size is fixed instead.\n\n"
        "Output: region_id and silhouette scores; the report has the full partition-quality assessment."
    )

    def add_parameters(self):
        self.numeric_field(self.FIELDS, "Attribute fields regions should be homogeneous on", multiple=True)
        self.integer(self.K_REGIONS, "Number of regions (k)", 5, minimum=1)
        self.enum(self.METHOD, "Variant", [label for label, _ in METHODS], 1)

    def add_trailing_parameters(self):
        self.integer(self.RESTARTS, "Random starts", 10, minimum=1, maximum=500)
        self.integer(self.MIN_SIZE, "Minimum units per region", 1, minimum=1)
        self.integer(self.RANDOM_SEED, "Random seed", 42, minimum=0)

    def compute(self, source, parameters, context, feedback):
        names = self.field_list(parameters, self.FIELDS, context)
        self.require(names, "Select at least one attribute field.")
        k = self.parameterAsInt(parameters, self.K_REGIONS, context)
        method_label, method = METHODS[self.parameterAsEnum(parameters, self.METHOD, context)]
        restarts = self.parameterAsInt(parameters, self.RESTARTS, context)
        min_size = self.parameterAsInt(parameters, self.MIN_SIZE, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        fids, x, dropped = self.read_matrix(source, names, feedback)
        self.require(len(fids) >= k, f"Only {len(fids)} complete features for k = {k} regions.")
        self.require(k * min_size <= len(fids), "k times the minimum region size exceeds the number of units.")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo(f"AZP: {restarts} random starts, {method_label.lower()}...")
        try:
            res = azp(x, graph, k, restarts=restarts, seed=seed, method=method, min_size=min_size)
        except ValueError as exc:
            self.require(False, str(exc))
        labels = res["labels"]
        section, per_unit, _summary = quality_section(x, labels, graph, compactness=region_compactness(source, fids, labels))
        rows = {fid: {"region_id": int(labels[i]), "sil": per_unit["sil"][i], "bnd_sil": per_unit["bnd_sil"][i],
                      "path_sil": per_unit["path_sil"][i]} for i, fid in enumerate(fids)}
        kpis = kpi_card_row_html([
            {"label": "Regions", "value": str(res["k"])},
            {"label": "Final SSD", "value": fmt(res["ssd"])},
            {"label": "Median SSD of the random starts", "value": fmt(res["start_ssd_median"]),
             "sublabel": f"improvement {100 * (1 - res['ssd'] / res['start_ssd_median']):.0f}%" if res["start_ssd_median"] > 0 else ""},
            {"label": "Random starts", "value": str(restarts)},
        ])
        guidance = analyst_guidance_html(
            "AZP",
            "AZP returns k contiguous regions with locally minimal within-region heterogeneity; a different seed can give a "
            "different, similarly good partition.",
            ["Several seeds give similar SSD and similar boundaries.", "Mean silhouette is positive.",
             "k is fixed by the planning question, not chosen to make the map look good."],
            ["Basic AZP stops at the first local optimum - compare with the simulated-annealing variant.",
             "Very small minimum sizes let AZP carve single-unit regions around outliers.",
             "The objective sees only the attribute fields; compactness is reported, not optimised."],
            ["Max-p Regions when a minimum size rather than k is given", "REDCAP for a deterministic tree-based partition",
             "Regionalization Quality Assessment to compare partitions"],
            "Treat the result as one good zoning among several; adopt boundaries that are stable across seeds.",
        )
        subtitle = (f"Fields: <strong>{html.escape(', '.join(names))}</strong> | k = <strong>{k}</strong> | Variant: "
                    f"<strong>{html.escape(method_label)}</strong> | Weights: <strong>{html.escape(weight_label)}</strong>")
        drop = callout(f"{dropped} feature(s) with missing values were left out.") if dropped else ""
        return ToolResult(report_html=report_page(self.DISPLAY, subtitle, kpis + drop + section + guidance),
                          fields=[("region_id", INT), ("sil", DOUBLE), ("bnd_sil", DOUBLE), ("path_sil", DOUBLE)],
                          rows=rows, field_docs=dict(REGION_DOCS), style=("cluster_id", "region_id"),
                          layer_title="PlanX GeoStats AZP regions")
