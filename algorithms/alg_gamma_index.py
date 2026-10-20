# -*- coding: utf-8 -*-
"""Gamma Index of Spatial Association Processing algorithm."""
from __future__ import annotations

import html

from qgis.core import QgsProcessingParameterBoolean

from ..core.charts import histogram_svg, kpi_card_row_html
from ..core.local_stats_engines import GAMMA_OPERATIONS, gamma_index
from ..core.reporting import analyst_guidance_html
from ._local_common import SpatialStatBase, ToolResult, callout, fmt, html_table, report_page

OPERATIONS = [
    ("Cross-product  y_i * y_j", "c"),
    ("Squared difference  (y_i - y_j)^2", "s"),
    ("Absolute difference  |y_i - y_j|", "a"),
]


class GammaIndexAlgorithm(SpatialStatBase):
    FIELD = "FIELD"
    OPERATION = "OPERATION"
    STANDARDIZE = "STANDARDIZE"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"

    ALG_ID = "gamma_index"
    DISPLAY = "Gamma Index of Spatial Association"
    GROUP = "02 | Urban Pattern Scan"
    GROUP_ID = "planx_pattern_scan"
    HAS_LAYER_OUTPUT = False
    REPORT_LABEL = "Gamma index report"
    HELP = (
        "The general cross-product statistic of Hubert, Golledge & Costanzo (1981): Gamma = sum_ij w_ij a_ij, "
        "where w_ij says whether i and j are neighbours and a_ij how alike their values are. Moran's I, "
        "Geary's C and the join counts are all special cases; Gamma lets you choose the similarity measure "
        "directly:\n"
        "- Cross-product: positive autocorrelation shows as a LARGE Gamma\n"
        "- Squared difference (Geary-like) and absolute difference: positive autocorrelation shows as a "
        "SMALL Gamma; the absolute difference is less dominated by a few extreme pairs\n\n"
        "Weights are binary (a neighbour counts once). Inference is total randomisation: the values are "
        "shuffled over the locations and Gamma recomputed; the report shows where the observed value falls "
        "in that reference distribution and gives all three similarity measures side by side, so a "
        "conclusion that depends on the choice of measure is visible."
    )

    def add_parameters(self):
        self.numeric_field(self.FIELD, "Numeric field")
        self.enum(self.OPERATION, "Similarity measure (shown first in the report)", [label for label, _ in OPERATIONS], 0)
        self.addParameter(QgsProcessingParameterBoolean(self.STANDARDIZE, "Standardise the field to z-scores first", defaultValue=True))

    def add_trailing_parameters(self):
        self.integer(self.PERMUTATIONS, "Permutations", 999, minimum=99, maximum=99999)
        self.integer(self.RANDOM_SEED, "Random seed", 42, minimum=0)

    def compute(self, source, parameters, context, feedback):
        field = self.parameterAsString(parameters, self.FIELD, context)
        op_label, op = OPERATIONS[self.parameterAsEnum(parameters, self.OPERATION, context)]
        standardize = self.parameterAsBool(parameters, self.STANDARDIZE, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        fids, x, dropped = self.read_matrix(source, [field], feedback)
        self.require(len(fids) >= 5, "At least 5 features with a value are required.")
        self.require(x[:, 0].std() > 0, "The field is constant.")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        graph = graph.binary()
        results = {}
        for _label, code in OPERATIONS:
            feedback.pushInfo(f"Gamma ({GAMMA_OPERATIONS[code]})...")
            results[code] = gamma_index(x[:, 0], graph, code, standardize, permutations, seed)
        main = results[op]
        verdict = self._verdict(main, op)
        kpis = kpi_card_row_html([
            {"label": "Gamma", "value": fmt(main["gamma"]), "sublabel": op_label},
            {"label": "Permutation mean", "value": fmt(main["sim_mean"])},
            {"label": "z", "value": fmt(main["z"], 3)},
            {"label": "p (two-sided)", "value": fmt(main["p"], 3), "tone": "good" if main["p"] < 0.05 else "neutral"},
        ])
        hist = histogram_svg(main["simulated"].tolist(), bins=30, observed=main["gamma"], x_label="Gamma under randomisation",
                             title="Observed Gamma against its reference distribution")
        rows = []
        for label, code in OPERATIONS:
            r = results[code]
            rows.append([label, r["gamma"], r["sim_mean"], r["z"], r["p"], self._verdict(r, code)])
        table = html_table(["Similarity measure", "Gamma", "Permutation mean", "z", "p", "Reading"], rows)
        agree = len({self._direction(results[c], c) for _l, c in OPERATIONS if results[c]["p"] < 0.05})
        disagree = callout("The three similarity measures disagree on the direction of the significant result; the "
                           "conclusion depends on how similarity is defined - inspect the extreme pairs before reporting it.") \
            if agree > 1 else ""
        guidance = analyst_guidance_html(
            "Gamma index",
            "Gamma sums an attribute-similarity score over every pair of neighbours and asks whether that sum is unusual "
            "for a random arrangement of the same values.",
            [
                "The neighbourhood definition reflects the interaction you care about.",
                "The three measures point the same way.",
                "With the cross-product, the field is standardised (otherwise Gamma is dominated by the mean).",
            ],
            [
                "An unstandardised cross-product of a strictly positive field is large under any arrangement; read it "
                "against the permutation mean, never alone.",
                "Squared differences are sensitive to a few extreme pairs; compare with the absolute difference.",
            ],
            ["Global Moran's I (the standardised cross-product with row weights)", "Geary's C", "Join Count Statistics"],
            "Use Gamma to confirm that a pattern found with Moran's I does not depend on the cross-product form of "
            "similarity.",
        )
        subtitle = (f"Field: <strong>{html.escape(field)}</strong> | Units: <strong>{len(fids)}</strong> | "
                    f"Weights: <strong>{html.escape(weight_label)}</strong> (binary) | Standardised: "
                    f"<strong>{'yes' if standardize else 'no'}</strong> | Permutations: <strong>{permutations}</strong>")
        drop = callout(f"{dropped} feature(s) with missing values were left out.") if dropped else ""
        body = (kpis + drop + callout(html.escape(verdict), info=True)
                + f"<section><h2>Reference distribution</h2>{hist}</section>"
                + f"<section><h2>All three similarity measures</h2>{table}{disagree}</section>" + guidance)
        return ToolResult(report_html=report_page(self.DISPLAY, subtitle, body),
                          outputs={"GAMMA": main["gamma"], "P_VALUE": main["p"], "Z_SCORE": main["z"]})

    @staticmethod
    def _direction(result, code):
        larger = result["gamma"] > result["sim_mean"]
        positive = larger if code == "c" else not larger
        return "positive" if positive else "negative"

    def _verdict(self, result, code):
        if not result["p"] < 0.05:
            return "No significant spatial association (p >= 0.05)."
        direction = self._direction(result, code)
        if direction == "positive":
            return "Significant positive spatial association: neighbours are more alike than chance."
        return "Significant negative spatial association: neighbours are less alike than chance."
