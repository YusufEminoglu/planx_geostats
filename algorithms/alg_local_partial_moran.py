# -*- coding: utf-8 -*-
"""Partial and Conditional Local Moran Processing algorithm."""
from __future__ import annotations

import html
from collections import Counter

import numpy as np

from ..core.charts import donut_chart_svg, kpi_card_row_html
from ..core.local_pattern_audit import MULTIPLE_TESTING_OPTIONS, adjust_p_values
from ..core.local_stats_engines import moran_local_conditional, moran_local_partial, moran_local_perm
from ..core.reporting import analyst_guidance_html
from ..core.symbology import LISA_QUADRANT_STYLE
from ._local_common import permutation_resolution_warning, with_notes, DOUBLE, INT, STRING, SpatialStatBase, ToolResult, callout, fmt, html_table, report_page

MODES = [
    ("Conditional - Local Moran of the field after the controls are regressed out", "conditional"),
    ("Partial - the unit's share of the field-lag slope, holding the controls fixed", "partial"),
]
CLASSES = ["HH", "LL", "HL", "LH", "Not Significant"]


class LocalPartialMoranAlgorithm(SpatialStatBase):
    FIELD = "FIELD"
    CONTROLS = "CONTROLS"
    MODE = "MODE"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"

    ALG_ID = "local_partial_moran"
    DISPLAY = "Partial and Conditional Local Moran (Clusters Net of Covariates)"
    GROUP = "03 | Hot Spots and Spatial Outliers"
    GROUP_ID = "planx_hotspots_outliers"
    OUTPUT_LABEL = "Partial / conditional LISA output"
    REPORT_LABEL = "Partial / conditional LISA report"
    HELP = (
        "Does a cluster survive once you account for what you already know explains the field? A cluster of "
        "high rents may be nothing more than a cluster of central locations; these two statistics (Wolf "
        "2024, implemented in esda as MoranLocalPartial and MoranLocalConditional) ask what is left.\n\n"
        "- Conditional: the field is regressed on the control fields (OLS with intercept, all unit-scaled) "
        "and Local Moran's I is computed on the residual. A remaining HH cluster is high beyond what the "
        "controls predict, next to neighbours that are also high beyond prediction.\n"
        "- Partial: the regression of the field's spatial lag on the field and the controls is decomposed "
        "into each unit's contribution to the field's own coefficient - the local association between the "
        "field and its surroundings with the controls held fixed.\n\n"
        "Inference is two-sided conditional permutation with the FDR correction by default. The report runs "
        "the plain Local Moran's I on the same field and cross-tabulates the classes, so you see which "
        "clusters the controls explain away. Output fields: pm_assoc, pm_p, pm_padj, quadrant, pm_nbrs."
    )

    def add_parameters(self):
        self.numeric_field(self.FIELD, "Field to analyse")
        self.numeric_field(self.CONTROLS, "Control fields (covariates)", multiple=True)
        self.enum(self.MODE, "Statistic", [label for label, _ in MODES], 0)

    def add_trailing_parameters(self):
        self.integer(self.PERMUTATIONS, "Conditional permutations", 999, minimum=99, maximum=9999)
        self.integer(self.RANDOM_SEED, "Random seed", 42, minimum=0)
        self.enum(self.MULTIPLE_TESTING, "Multiple-testing correction", [label for label, _ in MULTIPLE_TESTING_OPTIONS], 1)

    def compute(self, source, parameters, context, feedback):
        field = self.parameterAsString(parameters, self.FIELD, context)
        controls = [c for c in self.field_list(parameters, self.CONTROLS, context) if c != field]
        self.require(controls, "Select at least one control field other than the analysed field.")
        mode_label, mode = MODES[self.parameterAsEnum(parameters, self.MODE, context)]
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        correction = MULTIPLE_TESTING_OPTIONS[self.parameterAsEnum(parameters, self.MULTIPLE_TESTING, context)][1]
        fids, x, dropped = self.read_matrix(source, [field] + controls, feedback)
        self.require(len(fids) > len(controls) + 10, "Too few complete features for this many controls.")
        y, X = x[:, 0], x[:, 1:]
        self.require(np.std(y) > 0, "The analysed field is constant.")
        rank = np.linalg.matrix_rank(np.column_stack([np.ones(len(y)), X]))
        self.require(rank == X.shape[1] + 1, "The control fields are collinear (one is a combination of the others).")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo(f"Computing the {mode} local Moran statistic...")
        if mode == "conditional":
            res = moran_local_conditional(y, X, graph, permutations, seed)
        else:
            res = moran_local_partial(y, X, graph, permutations, seed)
        plain = moran_local_perm(y, graph, permutations, seed)

        def classes_for(p_raw, labels):
            adj = adjust_p_values(np.nan_to_num(p_raw, nan=1.0), correction)
            adj = np.where(np.isfinite(p_raw), adj, np.nan)
            return adj, [lab if np.isfinite(q) and q < 0.05 else "Not Significant" for lab, q in zip(labels, adj)]

        p_adj, quad = classes_for(res["p"], res["label"])
        _plain_adj, plain_quad = classes_for(plain["p"], plain["quadrant"])
        rows = {fid: {"pm_assoc": res["association"][i], "pm_p": res["p"][i], "pm_padj": p_adj[i],
                      "quadrant": quad[i], "plain_quad": plain_quad[i], "pm_nbrs": int(graph.cardinality[i])}
                for i, fid in enumerate(fids)}
        fields = [("pm_assoc", DOUBLE), ("pm_p", DOUBLE), ("pm_padj", DOUBLE), ("quadrant", STRING),
                  ("plain_quad", STRING), ("pm_nbrs", INT)]
        docs = {
            "pm_assoc": f"{mode.capitalize()} local Moran statistic",
            "pm_p": "Two-sided conditional-permutation p-value",
            "pm_padj": f"p-value after {correction} correction",
            "quadrant": f"Class after accounting for {', '.join(controls)}",
            "plain_quad": "Class from the plain Local Moran's I of the same field",
            "pm_nbrs": "Neighbours used",
        }
        report = self._report(field, controls, mode, mode_label, weight_label, permutations, correction,
                              quad, plain_quad, res.get("controls_r2"), dropped)
        warn = permutation_resolution_warning(permutations, int(np.isfinite(res["p"]).sum()), correction)
        if warn:
            feedback.pushWarning(warn)
        report = with_notes(report, [warn])
        return ToolResult(report_html=report, fields=fields, rows=rows, field_docs=docs,
                          style=("lisa", "quadrant"), layer_title="PlanX GeoStats partial/conditional LISA output")

    def _report(self, field, controls, mode, mode_label, weight_label, permutations, correction,
                quad, plain_quad, r2, dropped):
        counts = Counter(quad)
        plain_counts = Counter(plain_quad)
        sig = len(quad) - counts["Not Significant"]
        plain_sig = len(plain_quad) - plain_counts["Not Significant"]
        cards = [
            {"label": "Significant after controls", "value": str(sig)},
            {"label": "Significant, plain LISA", "value": str(plain_sig)},
            {"label": "Explained away", "value": str(sum(1 for a, b in zip(plain_quad, quad)
                                                       if a != "Not Significant" and b == "Not Significant")),
             "tone": "warn"},
        ]
        if r2 is not None and np.isfinite(r2):
            cards.append({"label": "Controls R²", "value": fmt(r2, 3), "sublabel": "share of the field they explain"})
        kpis = kpi_card_row_html(cards)
        colors = {label: color for _code, color, label in LISA_QUADRANT_STYLE}
        donut = donut_chart_svg([label for _c, _col, label in LISA_QUADRANT_STYLE],
                                [counts.get(code, 0) for code, _col, _l in LISA_QUADRANT_STYLE],
                                colors=colors, title="Classes net of the controls")
        cross = Counter(zip(plain_quad, quad))
        table = html_table(["Plain LISA \\ net of controls"] + CLASSES,
                           [[r] + [cross.get((r, c), 0) for c in CLASSES] for r in CLASSES])
        partial_note = ""
        if mode == "partial":
            partial_note = callout("Partial labels describe the unit's contribution to the partial slope (left component, "
                                   "centred) against the sign of its spatial lag; they are not the sign of the raw value.",
                                   info=True)
        guidance = analyst_guidance_html(
            "Partial / conditional local Moran",
            "A cluster that remains after the controls is spatial structure the controls do not account for - an "
            "omitted factor, a local policy effect or a spillover.",
            [
                "The controls are plausible causes or confounders of the field, measured on the same units.",
                "The controls are not themselves consequences of the field (that would remove real signal).",
                "Significant units still form contiguous groups.",
            ],
            [
                "Many controls on few units overfit: the residual becomes small and clusters vanish for statistical rather "
                "than substantive reasons.",
                "A cluster that disappears is explained in the regression sense, not proven to be caused by the controls.",
                "Non-linear effects of a control are not removed by a linear regression.",
            ],
            [
                "OLS Regression and LM Diagnostics to model the field with the same controls",
                "Spatial Lag or Spatial Error Regression if residual clustering is strong",
                "Local Moran's I (LISA) for the unadjusted pattern",
            ],
            "Prioritise clusters that survive adjustment: they are the places where something beyond the measured "
            "drivers is at work.",
        )
        subtitle = (f"Field: <strong>{html.escape(field)}</strong> | Controls: <strong>{html.escape(', '.join(controls))}</strong> | "
                    f"Mode: <strong>{html.escape(mode_label)}</strong> | Weights: <strong>{html.escape(weight_label)}</strong> | "
                    f"Permutations: <strong>{permutations}</strong>")
        drop = callout(f"{dropped} feature(s) with missing values were left out.") if dropped else ""
        body = (kpis + drop + partial_note + f"<section><h2>Classes</h2>{donut}</section>"
                + f"<section><h2>What the controls change</h2>{table}</section>" + guidance)
        return report_page(self.DISPLAY, subtitle, body)
