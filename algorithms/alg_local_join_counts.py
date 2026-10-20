# -*- coding: utf-8 -*-
"""Local Join Count Statistics (Anselin & Li 2019) Processing algorithm."""
from __future__ import annotations

import html
from collections import Counter

import numpy as np

from ..core.charts import donut_chart_svg, kpi_card_row_html
from ..core.local_pattern_audit import MULTIPLE_TESTING_OPTIONS, adjust_p_values
from ..core.local_stats_engines import local_join_counts
from ..core.reporting import analyst_guidance_html
from ._local_common import permutation_resolution_warning, with_notes, DOUBLE, INT, STRING, SpatialStatBase, ToolResult, callout, html_table, report_page

MODES = [
    ("Univariate - clusters of 1s (BB joins)", "univariate"),
    ("Bivariate co-location - units where both A and B are 1, next to units where both are 1", "clc"),
    ("Bivariate joins - units with A only, next to units with B only", "bjc"),
    ("Multivariate co-location - all selected variables are 1 together", "multivariate"),
]
RULES = [
    ("Fields are already 0/1", "binary"),
    ("1 where the value is above the field median", "median"),
    ("1 where the value is at or above the threshold below", "threshold"),
]
STYLE = [
    ("Significant cluster", "#b2182b", "Significant join cluster"),
    ("Not Significant", "#fddbc7", "Eligible, not significant"),
    ("Not eligible", "#f7f7f7", "Not eligible (focal condition not met)"),
]


def binarize(values, rule, threshold):
    v = np.asarray(values, dtype=float)
    if rule == "binary":
        return v
    if rule == "median":
        return (v > np.median(v)).astype(float)
    return (v >= threshold).astype(float)


class LocalJoinCountsAlgorithm(SpatialStatBase):
    FIELD_A = "FIELD_A"
    FIELD_B = "FIELD_B"
    EXTRA_FIELDS = "EXTRA_FIELDS"
    MODE = "MODE"
    RULE = "RULE"
    THRESHOLD = "THRESHOLD"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"

    ALG_ID = "local_join_counts"
    DISPLAY = "Local Join Count Statistics (Binary Clusters and Co-location)"
    GROUP = "03 | Hot Spots and Spatial Outliers"
    GROUP_ID = "planx_hotspots_outliers"
    OUTPUT_LABEL = "Local join count output"
    REPORT_LABEL = "Local join count report"
    HELP = (
        "Local join counts (Anselin & Li 2019) for presence/absence data - vacant or occupied, flooded or "
        "not, served or unserved. For each unit that has the property, the statistic counts how many of its "
        "neighbours have it too, and a one-sided conditional permutation test asks whether that count is "
        "larger than chance would give. The global Join Count Statistics tool says whether 1s cluster "
        "somewhere; this tool says where.\n\n"
        "Modes:\n"
        "- Univariate: clusters of 1s in field A\n"
        "- Bivariate co-location (CLC): units where A and B are both 1, surrounded by units where both are 1\n"
        "- Bivariate joins (BJC): units with A but not B whose neighbours have B but not A - two "
        "conditions sitting next to each other without overlapping\n"
        "- Multivariate co-location: every selected field is 1 at the unit and at its neighbours\n\n"
        "Fields can be 0/1 already, or be turned into 0/1 by the median or a threshold. Weights are used as "
        "binary (a neighbour is a neighbour). The statistic is only defined where the focal unit meets its "
        "condition; elsewhere it is written as 0 with a NULL p-value and the class 'Not eligible'.\n"
        "Output fields: ljc, ljc_p, ljc_padj, ljc_class, ljc_nbrs."
    )

    def add_parameters(self):
        self.numeric_field(self.FIELD_A, "Field A")
        self.enum(self.MODE, "Join type", [label for label, _ in MODES], 0)
        self.numeric_field(self.FIELD_B, "Field B (bivariate modes)", optional=True)
        self.numeric_field(self.EXTRA_FIELDS, "Further fields (multivariate mode)", optional=True, multiple=True)
        self.enum(self.RULE, "How to read the fields as 0/1", [label for label, _ in RULES], 0)
        self.double(self.THRESHOLD, "Threshold (when reading by threshold)", 1.0)

    def add_trailing_parameters(self):
        self.integer(self.PERMUTATIONS, "Conditional permutations", 999, minimum=99, maximum=9999)
        self.integer(self.RANDOM_SEED, "Random seed", 42, minimum=0)
        self.enum(self.MULTIPLE_TESTING, "Multiple-testing correction", [label for label, _ in MULTIPLE_TESTING_OPTIONS], 1)

    def compute(self, source, parameters, context, feedback):
        field_a = self.parameterAsString(parameters, self.FIELD_A, context)
        field_b = self.parameterAsString(parameters, self.FIELD_B, context)
        extra = self.field_list(parameters, self.EXTRA_FIELDS, context) or []
        mode_label, mode = MODES[self.parameterAsEnum(parameters, self.MODE, context)]
        rule = RULES[self.parameterAsEnum(parameters, self.RULE, context)][1]
        threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        correction = MULTIPLE_TESTING_OPTIONS[self.parameterAsEnum(parameters, self.MULTIPLE_TESTING, context)][1]

        names = [field_a]
        if mode in ("clc", "bjc"):
            self.require(bool(field_b), "This join type needs Field B.")
            names.append(field_b)
        elif mode == "multivariate":
            names += [f for f in ([field_b] if field_b else []) + list(extra) if f and f != field_a]
            self.require(len(names) >= 2, "Multivariate co-location needs at least two fields (Field B or Further fields).")
        fids, raw, dropped = self.read_matrix(source, names, feedback)
        self.require(len(fids) >= 5, "At least 5 features with values in every selected field are required.")
        cols = [binarize(raw[:, k], rule, threshold) for k in range(raw.shape[1])]
        if rule == "binary":
            for name, col in zip(names, cols):
                bad = sorted(set(np.unique(col)) - {0.0, 1.0})
                self.require(not bad, f"Field '{name}' is not 0/1 (found values such as {bad[:4]}). "
                                      "Choose the median or threshold rule, or recode the field.")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo("Counting local joins and running the permutation test...")
        res = local_join_counts(cols[0], graph, permutations, seed,
                                z=cols[1] if mode in ("clc", "bjc") else None,
                                mode=mode, extra=cols[1:] if mode == "multivariate" else None)
        eligible = res["eligible"]
        p = res["p"]
        p_adj = np.full(len(p), np.nan)
        tested = np.isfinite(p)
        if tested.any():
            p_adj[tested] = adjust_p_values(p[tested], correction)
        classes = []
        for i in range(len(fids)):
            if not eligible[i]:
                classes.append("Not eligible")
            elif np.isfinite(p_adj[i]) and p_adj[i] < 0.05:
                classes.append("Significant cluster")
            else:
                classes.append("Not Significant")
        rows = {fid: {"ljc": res["ljc"][i], "ljc_p": p[i], "ljc_padj": p_adj[i], "ljc_class": classes[i],
                      "ljc_nbrs": int(graph.cardinality[i])} for i, fid in enumerate(fids)}
        fields = [("ljc", DOUBLE), ("ljc_p", DOUBLE), ("ljc_padj", DOUBLE), ("ljc_class", STRING), ("ljc_nbrs", INT)]
        docs = {
            "ljc": "Local join count: neighbours meeting the join condition",
            "ljc_p": "One-sided conditional-permutation p-value (NULL where not tested)",
            "ljc_padj": f"p-value after {correction} correction across tested units",
            "ljc_class": "Significant cluster / Not Significant / Not eligible",
            "ljc_nbrs": "Neighbours (binary weights)",
        }
        prevalence = [float(c.mean()) for c in cols]
        report = self._report(names, mode_label, rule, threshold, prevalence, classes, res["ljc"], eligible,
                              weight_label, permutations, correction, dropped)
        warn = permutation_resolution_warning(permutations, int(tested.sum()), correction, two_sided=False)
        if warn:
            feedback.pushWarning(warn)
        report = with_notes(report, [warn])
        return ToolResult(report_html=report, fields=fields, rows=rows, field_docs=docs,
                          style=("categorized", "ljc_class", STYLE),
                          layer_title="PlanX GeoStats local join count output")

    def _report(self, names, mode_label, rule, threshold, prevalence, classes, ljc, eligible, weight_label,
                permutations, correction, dropped):
        counts = Counter(classes)
        kpis = kpi_card_row_html([
            {"label": "Eligible units", "value": str(int(eligible.sum())), "sublabel": f"of {len(classes)}"},
            {"label": "Significant clusters", "value": str(counts["Significant cluster"]),
             "tone": "good" if counts["Significant cluster"] else "neutral"},
            {"label": "Mean joins at eligible units", "value": f"{float(ljc[eligible].mean()):.2f}" if eligible.any() else "n/a"},
        ])
        donut = donut_chart_svg([label for _v, _c, label in STYLE], [counts.get(v, 0) for v, _c, _l in STYLE],
                                colors={label: c for _v, c, label in STYLE}, title="Local join classes")
        rule_text = {"binary": "fields read as 0/1", "median": "1 above each field's median",
                     "threshold": f"1 at or above {threshold:g}"}[rule]
        prev = html_table(["Field", "Share of units coded 1"], [[n, f"{100 * p:.1f}%"] for n, p in zip(names, prevalence)])
        rare = ""
        if eligible.sum() < 10:
            rare = callout("Fewer than ten units meet the focal condition. With so few candidates the test has little "
                           "power, and a single join can decide significance.")
        guidance = analyst_guidance_html(
            "Local join counts",
            "A significant unit has the property and is surrounded by more units with it than a random arrangement of "
            "the same number of 1s would give.",
            [
                "The 0/1 coding means something on the ground (a threshold you can defend, not an arbitrary cut).",
                "The global Join Count Statistics test already shows clustering.",
                "Several neighbouring units are significant together.",
            ],
            [
                "Median coding forces half the units to be 1; the result then describes 'above-median' clustering only.",
                "Very common properties (most units coded 1) leave little room for a test - nearly every neighbourhood is full.",
                "Units that do not meet the focal condition are 'Not eligible', not evidence of absence.",
            ],
            [
                "Join Count Statistics for the global test",
                "Colocation Quotient for point categories",
                "Local Moran's I on the underlying continuous field",
            ],
            "Use significant clusters to delineate contiguous areas of a binary condition (e.g. vacancy pockets) and "
            "the bivariate modes to find where two conditions coincide or sit side by side.",
        )
        subtitle = (f"Join type: <strong>{html.escape(mode_label)}</strong> | Coding: <strong>{html.escape(rule_text)}</strong> | "
                    f"Weights: <strong>{html.escape(weight_label)}</strong> (binary) | Permutations: <strong>{permutations}</strong> | "
                    f"Multiple testing: <strong>{html.escape(correction)}</strong>")
        drop = callout(f"{dropped} feature(s) with missing values were left out.") if dropped else ""
        body = kpis + drop + rare + f"<section><h2>Classes</h2>{donut}{prev}</section>" + guidance
        return report_page(self.DISPLAY, subtitle, body)
