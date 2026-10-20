# -*- coding: utf-8 -*-
"""Multivariate Local Geary (Anselin 2019) Processing algorithm."""
from __future__ import annotations

import html
from collections import Counter

import numpy as np

from ..core.charts import bar_chart_svg, donut_chart_svg, kpi_card_row_html
from ..core.local_pattern_audit import MULTIPLE_TESTING_OPTIONS, adjust_p_values
from ..core.local_stats_engines import multivariate_local_geary
from ..core.reporting import analyst_guidance_html
from ._local_common import permutation_resolution_warning, with_notes, DOUBLE, INT, STRING, SpatialStatBase, ToolResult, callout, fmt, html_table, report_page

STYLE = [
    ("Positive association", "#b2182b", "Positive association (similar to neighbours)"),
    ("Negative association", "#2166ac", "Negative association (multivariate outlier)"),
    ("Not Significant", "#f7f7f7", "Not Significant"),
    ("No neighbours", "#d9d9d9", "No neighbours"),
]


class MultivariateLocalGearyAlgorithm(SpatialStatBase):
    FIELDS = "FIELDS"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"

    ALG_ID = "multivariate_local_geary"
    DISPLAY = "Multivariate Local Geary (Cluster Analysis on Several Fields)"
    GROUP = "03 | Hot Spots and Spatial Outliers"
    GROUP_ID = "planx_hotspots_outliers"
    OUTPUT_LABEL = "Multivariate local Geary output"
    REPORT_LABEL = "Multivariate local Geary report"
    HELP = (
        "Anselin's (2019) multivariate Local Geary: is a unit similar to its neighbours across several "
        "attributes at once? Each field is z-scored and the statistic is the average, over fields, of the "
        "weighted squared differences between the unit and its neighbours:\n"
        "c_i = (1/k) sum_v sum_j w_ij (z_vi - z_vj)^2\n\n"
        "A significantly small c_i is positive local association - the unit and its neighbours form a "
        "multivariate cluster; a significantly large one is negative association - the unit stands out from "
        "its surroundings on the combination of fields. Unlike running Local Moran's I field by field, this "
        "finds places that are similar in the overall profile even when no single field is extreme.\n\n"
        "Inference: two-sided conditional permutation, false-discovery-rate correction on by default "
        "(Anselin recommends a strict cut-off because the test is run once per unit). Output fields: mlg_c, "
        "mlg_p, mlg_padj, mlg_class, mlg_nbrs. The report profiles the positive clusters field by field."
    )

    def add_parameters(self):
        self.numeric_field(self.FIELDS, "Numeric fields (two or more)", multiple=True)

    def add_trailing_parameters(self):
        self.integer(self.PERMUTATIONS, "Conditional permutations", 999, minimum=99, maximum=9999)
        self.integer(self.RANDOM_SEED, "Random seed", 42, minimum=0)
        self.enum(self.MULTIPLE_TESTING, "Multiple-testing correction", [label for label, _ in MULTIPLE_TESTING_OPTIONS], 1)

    def compute(self, source, parameters, context, feedback):
        names = self.field_list(parameters, self.FIELDS, context)
        self.require(len(names) >= 2, "Select at least two numeric fields.")
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        correction = MULTIPLE_TESTING_OPTIONS[self.parameterAsEnum(parameters, self.MULTIPLE_TESTING, context)][1]
        fids, x, dropped = self.read_matrix(source, names, feedback)
        self.require(len(fids) >= 10, "At least 10 features with values in every selected field are required.")
        constant = [n for n, v in zip(names, x.T) if np.std(v) == 0]
        self.require(not constant, f"These fields are constant and carry no information: {', '.join(constant)}")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo("Computing the multivariate local Geary statistic...")
        res = multivariate_local_geary(x, graph, permutations, seed)
        p_adj = adjust_p_values(np.nan_to_num(res["p"], nan=1.0), correction)
        p_adj = np.where(np.isfinite(res["p"]), p_adj, np.nan)
        classes = []
        for i in range(len(fids)):
            if graph.cardinality[i] == 0:
                classes.append("No neighbours")
            elif np.isfinite(p_adj[i]) and p_adj[i] < 0.05:
                classes.append(f"{res['association'][i]} association")
            else:
                classes.append("Not Significant")
        rows = {
            fid: {"mlg_c": res["c"][i], "mlg_p": res["p"][i], "mlg_padj": p_adj[i],
                  "mlg_class": classes[i], "mlg_nbrs": int(graph.cardinality[i])}
            for i, fid in enumerate(fids)
        }
        fields = [("mlg_c", DOUBLE), ("mlg_p", DOUBLE), ("mlg_padj", DOUBLE), ("mlg_class", STRING), ("mlg_nbrs", INT)]
        docs = {
            "mlg_c": "Multivariate local Geary statistic (mean over fields of weighted squared z-differences)",
            "mlg_p": "Two-sided conditional-permutation p-value",
            "mlg_padj": f"p-value after {correction} correction",
            "mlg_class": "Positive / Negative association or Not Significant",
            "mlg_nbrs": "Neighbours used",
        }
        report = self._report(names, x, classes, weight_label, permutations, correction, dropped)
        warn = permutation_resolution_warning(permutations, int(np.isfinite(res["p"]).sum()), correction)
        if warn:
            feedback.pushWarning(warn)
        report = with_notes(report, [warn])
        return ToolResult(report_html=report, fields=fields, rows=rows, field_docs=docs,
                          style=("categorized", "mlg_class", STYLE),
                          layer_title="PlanX GeoStats multivariate local Geary output")

    def _report(self, names, x, classes, weight_label, permutations, correction, dropped):
        counts = Counter(classes)
        n = len(classes)
        kpis = kpi_card_row_html([
            {"label": "Units", "value": str(n), "sublabel": f"{dropped} dropped for missing values"},
            {"label": "Positive association", "value": str(counts["Positive association"]), "tone": "good"},
            {"label": "Negative association", "value": str(counts["Negative association"]), "tone": "warn"},
            {"label": "Fields", "value": str(len(names))},
        ])
        donut = donut_chart_svg([label for _v, _c, label in STYLE], [counts.get(v, 0) for v, _c, _l in STYLE],
                                colors={label: c for _v, c, label in STYLE}, title="Classes")
        z = (x - x.mean(axis=0)) / x.std(axis=0)
        profile_rows = []
        pos = np.array([c == "Positive association" for c in classes])
        neg = np.array([c == "Negative association" for c in classes])
        for k, name in enumerate(names):
            profile_rows.append([name,
                                 float(z[pos, k].mean()) if pos.any() else float("nan"),
                                 float(z[neg, k].mean()) if neg.any() else float("nan")])
        profile = html_table(["Field", "Mean z in positive clusters", "Mean z at negative units"], profile_rows)
        chart = ""
        if pos.any():
            chart = bar_chart_svg(names, [float(z[pos, k].mean()) for k in range(len(names))],
                                  title="Profile of the positive-association units (mean z-score per field)")
        corr = np.corrcoef(x.T)
        high_corr = [(names[a], names[b], corr[a, b]) for a in range(len(names)) for b in range(a + 1, len(names))
                     if abs(corr[a, b]) > 0.8]
        corr_note = ""
        if high_corr:
            corr_note = callout("Strongly correlated fields count twice in the statistic: " + "; ".join(
                f"{html.escape(a)} / {html.escape(b)} (r = {fmt(r, 2)})" for a, b, r in high_corr)
                + ". Consider keeping one of each pair.")
        guidance = analyst_guidance_html(
            "Multivariate local Geary",
            "A positive-association unit resembles its neighbours across the selected fields taken together; a "
            "negative-association unit differs from its neighbours on that combination.",
            [
                "The fields describe one concept you want to find similar places on (e.g. a deprivation profile).",
                "No pair of fields is so correlated that one dimension is counted twice.",
                "Significant units group together rather than appearing alone.",
            ],
            [
                "Positive association does not say high or low - read the profile table to see what the cluster shares.",
                "Every field gets equal weight after z-scoring; an irrelevant field dilutes the signal.",
                "With many fields the statistic tends towards its mean; keep the set focused.",
            ],
            [
                "Local Moran's I on each field to see which one drives a cluster",
                "SKATER Regionalization or Max-p Regions to turn similar neighbourhoods into zones",
                "Multivariate Clustering for the same fields without the spatial constraint",
            ],
            "Use positive-association zones as candidate homogeneous planning areas and negative-association units as "
            "places whose combination of conditions breaks from the surroundings.",
        )
        subtitle = (f"Fields: <strong>{html.escape(', '.join(names))}</strong> | Weights: <strong>{html.escape(weight_label)}</strong> | "
                    f"Permutations: <strong>{permutations}</strong> | Multiple testing: <strong>{html.escape(correction)}</strong>")
        body = (kpis + corr_note + f"<section><h2>Classes</h2>{donut}</section>"
                + f"<section><h2>What the clusters share</h2>{chart}{profile}</section>" + guidance)
        return report_page(self.DISPLAY, subtitle, body)
