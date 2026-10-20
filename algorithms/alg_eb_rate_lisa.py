# -*- coding: utf-8 -*-
"""Rate Cluster and Outlier Analysis (EB Local Moran) Processing algorithm."""
from __future__ import annotations

import html
from collections import Counter

import numpy as np

from qgis.core import QgsProcessingParameterBoolean

from ..core.charts import donut_chart_svg, kpi_card_row_html
from ..core.local_pattern_audit import MULTIPLE_TESTING_OPTIONS, adjust_p_values
from ..core.local_stats_engines import assuncao_reis_rate, moran_global_perm, moran_local_perm
from ..core.reporting import analyst_guidance_html
from ..core.symbology import LISA_QUADRANT_STYLE
from ._local_common import permutation_resolution_warning, with_notes, DOUBLE, INT, STRING, SpatialStatBase, ToolResult, callout, fmt, html_table, report_page

CLASSES = ["HH", "LL", "HL", "LH", "Not Significant"]


def classify(quadrants, p_adjusted, alpha=0.05):
    return [q if np.isfinite(p) and p < alpha else "Not Significant" for q, p in zip(quadrants, p_adjusted)]


class EBRateLisaAlgorithm(SpatialStatBase):
    EVENTS = "EVENTS"
    BASE = "BASE"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"
    COMPARE_RAW = "COMPARE_RAW"

    ALG_ID = "eb_rate_lisa"
    DISPLAY = "Rate Cluster and Outlier Analysis (EB Local Moran)"
    GROUP = "03 | Hot Spots and Spatial Outliers"
    GROUP_ID = "planx_hotspots_outliers"
    OUTPUT_LABEL = "EB rate LISA output"
    REPORT_LABEL = "EB rate LISA report"
    HELP = (
        "Local Moran's I for rates (Assuncao & Reis 1999; Anselin, Lozano-Gracia & Koschinsky 2006). "
        "Running a LISA on a raw rate lets small denominators manufacture clusters: a unit with a tiny "
        "population swings to an extreme rate by chance and drags its neighbourhood into an HH or LL class. "
        "This tool first converts events and base into the Assuncao-Reis empirical-Bayes standardised rate "
        "- each rate's departure from the study-area rate divided by its own estimated standard error - and "
        "runs Local Moran's I on that.\n\n"
        "Inference is conditional permutation (two-sided), with the false-discovery-rate correction on by "
        "default. With 'Compare with raw-rate LISA' on, the same test is also run on the raw rate and the "
        "report cross-tabulates the two classifications, so the clusters that exist only because of small "
        "numbers are named.\n\n"
        "Output fields: rate_raw, eb_z (EB-standardised rate), lisa_i, lisa_p, lisa_padj, quadrant "
        "(HH/LL/HL/LH/Not Significant), raw_quad (raw-rate class, when compared) and lisa_nbrs."
    )

    def add_parameters(self):
        self.numeric_field(self.EVENTS, "Event count field (numerator)")
        self.numeric_field(self.BASE, "Population at risk field (denominator)")

    def add_trailing_parameters(self):
        self.integer(self.PERMUTATIONS, "Conditional permutations", 999, minimum=99, maximum=9999)
        self.integer(self.RANDOM_SEED, "Random seed", 42, minimum=0)
        self.enum(self.MULTIPLE_TESTING, "Multiple-testing correction", [label for label, _ in MULTIPLE_TESTING_OPTIONS], 1)
        self.addParameter(QgsProcessingParameterBoolean(self.COMPARE_RAW, "Compare with raw-rate LISA", defaultValue=True))

    def compute(self, source, parameters, context, feedback):
        events_field = self.parameterAsString(parameters, self.EVENTS, context)
        base_field = self.parameterAsString(parameters, self.BASE, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        correction = MULTIPLE_TESTING_OPTIONS[self.parameterAsEnum(parameters, self.MULTIPLE_TESTING, context)][1]
        compare = self.parameterAsBool(parameters, self.COMPARE_RAW, context)

        fids, matrix, dropped = self.read_matrix(source, [events_field, base_field], feedback)
        keep = (matrix[:, 1] > 0) & (matrix[:, 0] >= 0) if len(fids) else np.zeros(0, bool)
        fids = [fid for fid, k in zip(fids, keep) if k]
        matrix = matrix[keep]
        self.require(len(fids) >= 10, "At least 10 features with a valid count and a positive base are required.")
        events, base = matrix[:, 0], matrix[:, 1]
        self.require(events.sum() > 0, "The event field sums to zero.")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)

        raw = events / base
        ebz = assuncao_reis_rate(events, base)
        feedback.pushInfo("Global Moran's I on the raw and the EB-standardised rate...")
        g_eb = moran_global_perm(ebz, graph, permutations, seed)
        g_raw = moran_global_perm(raw, graph, permutations, seed)
        feedback.pushInfo("Local Moran's I on the EB-standardised rate (conditional permutation)...")
        loc = moran_local_perm(ebz, graph, permutations, seed)
        p_adj = adjust_p_values(np.nan_to_num(loc["p"], nan=1.0), correction)
        p_adj = np.where(np.isfinite(loc["p"]), p_adj, np.nan)
        quad = classify(loc["quadrant"], p_adj)
        raw_quad = None
        if compare:
            feedback.pushInfo("Same test on the raw rate for comparison...")
            loc_raw = moran_local_perm(raw, graph, permutations, seed)
            p_raw = adjust_p_values(np.nan_to_num(loc_raw["p"], nan=1.0), correction)
            raw_quad = classify(loc_raw["quadrant"], np.where(np.isfinite(loc_raw["p"]), p_raw, np.nan))

        rows = {}
        for i, fid in enumerate(fids):
            row = {
                "rate_raw": raw[i], "eb_z": ebz[i], "lisa_i": loc["I"][i], "lisa_p": loc["p"][i],
                "lisa_padj": p_adj[i], "quadrant": quad[i], "lisa_nbrs": int(graph.cardinality[i]),
            }
            if raw_quad is not None:
                row["raw_quad"] = raw_quad[i]
            rows[fid] = row
        fields = [("rate_raw", DOUBLE), ("eb_z", DOUBLE), ("lisa_i", DOUBLE), ("lisa_p", DOUBLE),
                  ("lisa_padj", DOUBLE), ("quadrant", STRING), ("lisa_nbrs", INT)]
        if raw_quad is not None:
            fields.insert(6, ("raw_quad", STRING))
        docs = {
            "rate_raw": "Raw rate (events / base)",
            "eb_z": "Assuncao-Reis EB-standardised rate",
            "lisa_i": "Local Moran's I of the EB-standardised rate",
            "lisa_p": "Two-sided conditional-permutation p-value",
            "lisa_padj": f"p-value after {correction} correction (used for the class)",
            "quadrant": "LISA class of the EB-standardised rate",
            "raw_quad": "LISA class the raw rate would have produced",
            "lisa_nbrs": "Neighbours used",
        }
        report = self._report(events_field, base_field, weight_label, correction, permutations, g_eb, g_raw,
                              quad, raw_quad, len(fids), dropped)
        warn = permutation_resolution_warning(permutations, int(np.isfinite(loc["p"]).sum()), correction)
        if warn:
            feedback.pushWarning(warn)
        report = with_notes(report, [warn])
        return ToolResult(report_html=report, fields=fields, rows=rows, field_docs=docs,
                          style=("lisa", "quadrant"), layer_title="PlanX GeoStats EB rate LISA output")

    def _report(self, events_field, base_field, weight_label, correction, permutations, g_eb, g_raw,
                quad, raw_quad, n, dropped):
        counts = Counter(quad)
        clusters = counts["HH"] + counts["LL"]
        outliers = counts["HL"] + counts["LH"]
        kpis = kpi_card_row_html([
            {"label": "Moran's I, EB rate", "value": fmt(g_eb["I"], 3), "sublabel": f"p = {fmt(g_eb['p'], 3)}",
             "tone": "good" if g_eb["p"] < 0.05 else "neutral"},
            {"label": "Moran's I, raw rate", "value": fmt(g_raw["I"], 3), "sublabel": f"p = {fmt(g_raw['p'], 3)}"},
            {"label": "Cluster units", "value": str(clusters), "sublabel": "HH + LL"},
            {"label": "Outlier units", "value": str(outliers), "sublabel": "HL + LH"},
        ])
        colors = {label: color for _code, color, label in LISA_QUADRANT_STYLE}
        donut = donut_chart_svg([label for _c, _col, label in LISA_QUADRANT_STYLE],
                                [counts.get(code, 0) for code, _col, _l in LISA_QUADRANT_STYLE],
                                colors=colors, title="EB rate LISA classes")
        compare_html = ""
        if raw_quad is not None:
            cross = Counter(zip(raw_quad, quad))
            table_rows = [[r] + [cross.get((r, c), 0) for c in CLASSES] for r in CLASSES]
            only_raw = sum(cross.get((r, "Not Significant"), 0) for r in CLASSES if r != "Not Significant")
            only_eb = sum(cross.get(("Not Significant", c), 0) for c in CLASSES if c != "Not Significant")
            compare_html = (
                "<section><h2>Raw-rate LISA against EB-rate LISA</h2>"
                + html_table(["Raw-rate class \\ EB-rate class"] + CLASSES, table_rows)
                + callout(f"<strong>{only_raw}</strong> unit(s) are significant on the raw rate but not once the rate is "
                          f"stabilised - candidate small-number artefacts. <strong>{only_eb}</strong> unit(s) only become "
                          "significant after stabilisation, typically large-base units whose genuine difference was "
                          "masked by noisy small neighbours.", info=True)
                + "</section>"
            )
        drop_note = callout(f"{dropped} feature(s) were left out for a missing count or base.") if dropped else ""
        guidance = analyst_guidance_html(
            "EB rate LISA",
            "Each unit's EB-standardised rate is compared with the weighted average of its neighbours'. HH and LL are "
            "contiguous areas of genuinely high or low risk once the reliability of each rate is accounted for.",
            [
                "The global Moran's I on the EB rate is significant - local classes are then decomposing a real pattern.",
                "Significant units form contiguous groups rather than isolated cells.",
                "Events and base describe the same population and period.",
            ],
            [
                "A class that appears only in the raw-rate column of the cross-table is a likely small-number artefact.",
                "The EB standardisation assumes a Poisson-like count process; strongly over-dispersed counts need care.",
                "Permutation p-values are random to the order of 1/sqrt(permutations); re-run with another seed before "
                "acting on a borderline unit.",
            ],
            [
                "Rate Smoothing (Empirical Bayes) to map the smoothed rate itself",
                "Hot Spot Analysis (Getis-Ord Gi*) on the spatial EB rate",
                "Partial and Conditional Local Moran to check whether a covariate explains the clusters",
            ],
            "Target HH clusters for intervention and treat raw-only clusters as data-quality or monitoring questions "
            "rather than as evidence of elevated risk.",
        )
        subtitle = (f"Events: <strong>{html.escape(events_field)}</strong> | Base: <strong>{html.escape(base_field)}</strong> | "
                    f"Units: <strong>{n}</strong> | Weights: <strong>{html.escape(weight_label)}</strong> | "
                    f"Permutations: <strong>{permutations}</strong> | Multiple testing: <strong>{html.escape(correction)}</strong>")
        body = kpis + drop_note + f"<section><h2>Class breakdown</h2>{donut}</section>" + compare_html + guidance
        return report_page(self.DISPLAY, subtitle, body)
