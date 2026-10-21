# -*- coding: utf-8 -*-
"""Max-p Regions Processing algorithm."""
from __future__ import annotations

import html

import numpy as np

from ..core.charts import kpi_card_row_html
from ..core.regionalization_engines import max_p_regions
from ..core.reporting import analyst_guidance_html
from ._local_common import DOUBLE, INT, SpatialStatBase, ToolResult, callout, fmt, report_page
from ._region_common import REGION_DOCS, quality_section, region_compactness

SEARCH = [
    ("Simulated annealing (Wei, Rey & Knaap 2020)", "sa"),
    ("Tabu search", "tabu"),
    ("Greedy improving moves only", "greedy"),
    ("None (construction phase only)", "none"),
]


class MaxPRegionsAlgorithm(SpatialStatBase):
    FIELDS = "FIELDS"
    FLOOR_FIELD = "FLOOR_FIELD"
    THRESHOLD = "THRESHOLD"
    ITERATIONS = "ITERATIONS"
    LOCAL_SEARCH = "LOCAL_SEARCH"
    RANDOM_SEED = "RANDOM_SEED"

    ALG_ID = "max_p_regions"
    DISPLAY = "Max-p Regions (Zones Meeting a Minimum Size)"
    GROUP = "03 | Hot Spots and Spatial Outliers"
    GROUP_ID = "planx_hotspots_outliers"
    OUTPUT_LABEL = "Max-p regions"
    REPORT_LABEL = "Max-p regions report"
    HELP = (
        "The planning question 'what is the largest number of contiguous districts I can draw so that every "
        "district has at least a given population (or area, or number of facilities)?' - answered by the "
        "max-p-regions problem of Duque, Anselin & Rey (2012). Unlike SKATER or K-means you do not choose the "
        "number of regions: you choose a floor, and the method finds as many regions as the floor allows, "
        "then makes them as internally similar as it can on the attribute fields.\n\n"
        "The solution is heuristic (the problem is NP-hard): a construction phase grows regions from random "
        "seeds many times and keeps the run with the most regions, then a local search (simulated annealing "
        "by default, after Wei, Rey & Knaap 2020) moves units across region borders while keeping every "
        "region contiguous and above the floor. Units in a part of the map that cannot reach the floor on "
        "its own are left unassigned (NULL).\n\n"
        "Output: region_id plus the unit's silhouette scores; the report gives the floor reached by every "
        "region and the full partition-quality assessment."
    )

    def add_parameters(self):
        self.numeric_field(self.FIELDS, "Attribute fields regions should be homogeneous on", multiple=True)
        self.numeric_field(self.FLOOR_FIELD, "Floor field (e.g. population) - summed per region")
        self.double(self.THRESHOLD, "Minimum floor per region", 10000.0, minimum=1e-9)

    def add_trailing_parameters(self):
        self.integer(self.ITERATIONS, "Construction runs (more = closer to the true maximum p)", 99, minimum=1, maximum=5000)
        self.enum(self.LOCAL_SEARCH, "Local search", [label for label, _ in SEARCH], 0)
        self.integer(self.RANDOM_SEED, "Random seed", 42, minimum=0)

    def compute(self, source, parameters, context, feedback):
        names = self.field_list(parameters, self.FIELDS, context)
        self.require(names, "Select at least one attribute field.")
        floor_field = self.parameterAsString(parameters, self.FLOOR_FIELD, context)
        threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)
        iterations = self.parameterAsInt(parameters, self.ITERATIONS, context)
        search_label, search = SEARCH[self.parameterAsEnum(parameters, self.LOCAL_SEARCH, context)]
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        fids, m, dropped = self.read_matrix(source, names + [floor_field], feedback)
        self.require(len(fids) >= 3, "At least 3 complete features are required.")
        x, floor = m[:, :-1], m[:, -1]
        self.require(np.all(floor >= 0), "The floor field must not contain negative values.")
        self.require(floor.sum() >= threshold, f"The whole layer sums to {floor.sum():g} on the floor field, below the threshold.")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo(f"Max-p: {iterations} construction runs, then {search_label.lower()}...")
        try:
            res = max_p_regions(x, graph, floor, threshold, iterations=iterations, seed=seed, local_search=search)
        except ValueError as exc:
            self.require(False, str(exc))
        labels = res["labels"]
        comp = region_compactness(source, fids, labels)
        floors = {r: res["floor_by_region"][r] for r in range(res["p"])}
        section, per_unit, summary = quality_section(
            x, labels, graph, compactness=comp, extra_region_cols={"Floor sum": {r: f"{v:,.0f}" for r, v in floors.items()}})
        rows = {fid: {"region_id": int(labels[i]) if labels[i] >= 0 else None, "sil": per_unit["sil"][i],
                      "bnd_sil": per_unit["bnd_sil"][i], "path_sil": per_unit["path_sil"][i]}
                for i, fid in enumerate(fids)}
        fields = [("region_id", INT), ("sil", DOUBLE), ("bnd_sil", DOUBLE), ("path_sil", DOUBLE)]
        kpis = kpi_card_row_html([
            {"label": "Regions (p)", "value": str(res["p"]), "sublabel": f"floor {threshold:,.0f} on {floor_field}"},
            {"label": "Smallest region floor", "value": f"{min(floors.values()):,.0f}"},
            {"label": "SSD after search", "value": fmt(res["ssd"]), "sublabel": f"construction {fmt(res['construction_ssd'])}"},
            {"label": "Unassigned units", "value": str(res["unassigned"]), "tone": "warn" if res["unassigned"] else "neutral"},
        ])
        upper = int(floor.sum() // threshold)
        note = callout(f"The floor caps p at {upper} (total {floor.sum():,.0f} / threshold {threshold:,.0f}); this run "
                       f"found {res['p']}. The gap is the price of contiguity and of the attribute objective, or a sign "
                       "that more construction runs would find more regions.", info=True)
        guidance = analyst_guidance_html(
            "Max-p regions",
            "Each region is contiguous, meets the floor, and is as homogeneous on the attribute fields as the heuristic "
            "could make it while keeping as many regions as possible.",
            ["The floor is a real requirement (minimum population for a service district, minimum sample for a statistic).",
             "Re-running with another seed gives the same p.", "Mean silhouette is positive."],
            ["The heuristic does not prove p is the maximum; compare several seeds and more construction runs.",
             "Unassigned units sit in parts of the map that cannot reach the floor on their own.",
             "Highly correlated attribute fields count the same dimension twice."],
            ["Regionalization Quality Assessment to compare this partition with SKATER, AZP or REDCAP",
             "S-maup to check whether this number of zones is robust", "AZP when you want a fixed number of regions"],
            "Use the regions as candidate service or reporting districts that each meet the minimum size; check the "
            "weakest-silhouette regions on the ground before adopting the boundaries.",
        )
        subtitle = (f"Fields: <strong>{html.escape(', '.join(names))}</strong> | Floor: <strong>{html.escape(floor_field)}</strong> "
                    f"&ge; <strong>{threshold:g}</strong> | Search: <strong>{html.escape(search_label)}</strong> | "
                    f"Weights: <strong>{html.escape(weight_label)}</strong>")
        drop = callout(f"{dropped} feature(s) with missing values were left out.") if dropped else ""
        body = kpis + drop + note + section + guidance
        docs = dict(REGION_DOCS)
        return ToolResult(report_html=report_page(self.DISPLAY, subtitle, body), fields=fields, rows=rows,
                          field_docs=docs, style=("cluster_id", "region_id"), layer_title="PlanX GeoStats max-p regions")
