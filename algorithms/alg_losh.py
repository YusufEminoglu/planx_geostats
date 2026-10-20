# -*- coding: utf-8 -*-
"""Local Spatial Heteroskedasticity (LOSH, Ord & Getis 2012) Processing algorithm."""
from __future__ import annotations

import html
from collections import Counter

import numpy as np

from qgis.core import QgsProcessingParameterBoolean

from ..core.charts import donut_chart_svg, kpi_card_row_html, scatter_plot_svg
from ..core.local_pattern_audit import MULTIPLE_TESTING_OPTIONS, adjust_p_values
from ..core.local_stats_engines import getis_ord_star, losh
from ..core.reporting import analyst_guidance_html
from ._local_common import DOUBLE, STRING, SpatialStatBase, ToolResult, callout, fmt, report_page

TYPOLOGY = [
    ("Hot spot, heterogeneous", "#b2182b", "Hot spot with heterogeneous surroundings"),
    ("Hot spot, homogeneous", "#ef8a62", "Hot spot with uniform surroundings"),
    ("Cold spot, heterogeneous", "#2166ac", "Cold spot with heterogeneous surroundings"),
    ("Cold spot, homogeneous", "#67a9cf", "Cold spot with uniform surroundings"),
    ("Heterogeneous, average level", "#7b3294", "High local variance at an average level"),
    ("Not significant", "#f7f7f7", "Not significant"),
]


def typology(gi_p, gi_z, h_p_adj, h_value, alpha=0.05):
    hot = gi_p < alpha and gi_z > 0
    cold = gi_p < alpha and gi_z < 0
    hetero = np.isfinite(h_p_adj) and h_p_adj < alpha and h_value > 1
    if hot:
        return "Hot spot, heterogeneous" if hetero else "Hot spot, homogeneous"
    if cold:
        return "Cold spot, heterogeneous" if hetero else "Cold spot, homogeneous"
    if hetero:
        return "Heterogeneous, average level"
    return "Not significant"


class LOSHAlgorithm(SpatialStatBase):
    FIELD = "FIELD"
    EXPONENT = "EXPONENT"
    INCLUDE_SELF = "INCLUDE_SELF"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"

    ALG_ID = "losh_heteroskedasticity"
    DISPLAY = "Local Spatial Heteroskedasticity (LOSH)"
    GROUP = "03 | Hot Spots and Spatial Outliers"
    GROUP_ID = "planx_hotspots_outliers"
    OUTPUT_LABEL = "LOSH output"
    REPORT_LABEL = "LOSH report"
    HELP = (
        "Ord & Getis's (2012) H_i measures how variable a field is inside each unit's neighbourhood, the "
        "counterpart of Gi*, which measures how high it is. H_i is the weighted mean of the local residuals "
        "|x_j - local mean|^a around the unit, divided by the study-area average of the same residuals, so "
        "H_i = 1 is ordinary variability, H_i > 1 a heterogeneous neighbourhood (a transition zone or a "
        "mixed area) and H_i < 1 a uniform one.\n\n"
        "The tool computes Gi* alongside and combines the two into Ord & Getis's typology: a hot spot "
        "with uniform surroundings is a coherent high-value area; a hot spot with heterogeneous "
        "surroundings is a high average produced by a mix of very high and ordinary values; high variance "
        "at an average level marks a boundary between regimes.\n\n"
        "Residuals are taken about each unit's own local mean - the definition implemented in both spdep "
        "and esda (the paper prints the residual about the focal unit's mean; the manual discusses the "
        "difference). Inference: the chi-square approximation of Ord & Getis, valid for exponent a = 2; "
        "for any other exponent the p-values are left NULL rather than reported as if valid.\n"
        "Output fields: losh_h, losh_p, losh_padj, local_mean, gi_z, losh_type."
    )

    def add_parameters(self):
        self.numeric_field(self.FIELD, "Numeric field")

    def add_trailing_parameters(self):
        self.double(self.EXPONENT, "Residual exponent a (2 = variance, 1 = absolute deviation)", 2.0, minimum=0.1, maximum=4.0)
        self.addParameter(QgsProcessingParameterBoolean(
            self.INCLUDE_SELF, "Include each unit in its own neighbourhood (Gi*-style window)", defaultValue=True))
        self.enum(self.MULTIPLE_TESTING, "Multiple-testing correction", [label for label, _ in MULTIPLE_TESTING_OPTIONS], 1)

    def compute(self, source, parameters, context, feedback):
        field = self.parameterAsString(parameters, self.FIELD, context)
        a = self.parameterAsDouble(parameters, self.EXPONENT, context)
        include_self = self.parameterAsBool(parameters, self.INCLUDE_SELF, context)
        correction = MULTIPLE_TESTING_OPTIONS[self.parameterAsEnum(parameters, self.MULTIPLE_TESTING, context)][1]
        fids, x, dropped = self.read_matrix(source, [field], feedback)
        self.require(len(fids) >= 10, "At least 10 features with a value are required.")
        y = x[:, 0]
        self.require(np.std(y) > 0, "The field is constant.")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo("Computing LOSH and Gi*...")
        try:
            res = losh(y, graph, a, include_self)
        except ValueError as exc:
            self.require(False, str(exc))
        gi = getis_ord_star(y, graph)
        p = res["p"]
        p_adj = np.full(len(p), np.nan)
        tested = np.isfinite(p)
        if tested.any():
            p_adj[tested] = adjust_p_values(p[tested], correction)
        gi_adj = adjust_p_values(gi["p"], correction)
        types = [typology(gi_adj[i], gi["z"][i], p_adj[i], res["H"][i]) for i in range(len(fids))]
        rows = {fid: {"losh_h": res["H"][i], "losh_p": p[i], "losh_padj": p_adj[i], "local_mean": res["local_mean"][i],
                      "gi_z": gi["z"][i], "losh_type": types[i]} for i, fid in enumerate(fids)}
        fields = [("losh_h", DOUBLE), ("losh_p", DOUBLE), ("losh_padj", DOUBLE), ("local_mean", DOUBLE),
                  ("gi_z", DOUBLE), ("losh_type", STRING)]
        docs = {
            "losh_h": "LOSH H_i (1 = study-area average local variability)",
            "losh_p": "Chi-square p-value for H_i (NULL unless a = 2)",
            "losh_padj": f"p-value after {correction} correction",
            "local_mean": "Local (window) mean of the field",
            "gi_z": "Getis-Ord Gi* z-score (binary weights, self included)",
            "losh_type": "Ord & Getis mean/variance typology",
        }
        report = self._report(field, a, include_self, weight_label, correction, res, gi, types, dropped)
        return ToolResult(report_html=report, fields=fields, rows=rows, field_docs=docs,
                          style=("categorized", "losh_type", TYPOLOGY), layer_title="PlanX GeoStats LOSH output")

    def _report(self, field, a, include_self, weight_label, correction, res, gi, types, dropped):
        counts = Counter(types)
        h = res["H"]
        kpis = kpi_card_row_html([
            {"label": "Units", "value": str(len(types)), "sublabel": f"{dropped} dropped"},
            {"label": "Median H", "value": fmt(float(np.median(h)), 3), "sublabel": "1 = average variability"},
            {"label": "Heterogeneous units", "value": str(sum(v for k, v in counts.items() if "heterogeneous" in k.lower())),
             "tone": "warn"},
            {"label": "Hot / cold spots", "value": f"{counts['Hot spot, heterogeneous'] + counts['Hot spot, homogeneous']}"
                                                   f" / {counts['Cold spot, heterogeneous'] + counts['Cold spot, homogeneous']}"},
        ])
        donut = donut_chart_svg([label for _v, _c, label in TYPOLOGY], [counts.get(v, 0) for v, _c, _l in TYPOLOGY],
                                colors={label: c for _v, c, label in TYPOLOGY}, title="Mean/variance typology")
        scatter = scatter_plot_svg(gi["z"].tolist(), h.tolist(), x_label="Gi* z (local level)", y_label="LOSH H (local variability)",
                                   title="Level against variability", trend_line=False, quadrant_shading=True,
                                   split_x=0.0, split_y=1.0)
        a_note = "" if abs(a - 2.0) < 1e-12 else callout(
            f"Exponent a = {a:g}: the chi-square inference is derived for a = 2 only, so p-values are NULL and the "
            "typology uses Gi* alone. Read H as a descriptive measure.")
        guidance = analyst_guidance_html(
            "LOSH",
            "H_i compares the spread of values in a unit's neighbourhood with the typical spread across the study area. "
            "Combined with Gi*, it separates uniform hot spots from hot spots made of mixed values.",
            [
                "The neighbourhood is wide enough to hold several values (a variance from two values is unstable).",
                "Exponent a = 2 when you need p-values.",
                "The field is roughly continuous; heavy ties make local variance degenerate.",
            ],
            [
                "A single extreme value inflates H for every unit whose window contains it - check the map for a ring "
                "of heterogeneous units around one outlier.",
                "H depends on the window: a changed weights definition changes which units look heterogeneous.",
                "The chi-square approximation is rough for small windows.",
            ],
            [
                "Hot Spot Analysis (Getis-Ord Gi*) for the full Gi* confidence classes",
                "Local Geary's C for local dissimilarity of a unit from its neighbours",
                "GW Summary Statistics for a kernel-weighted local standard deviation",
            ],
            "Heterogeneous hot spots call for finer-grained investigation before area-wide intervention; uniform hot "
            "spots can be treated as one coherent area.",
        )
        subtitle = (f"Field: <strong>{html.escape(field)}</strong> | a = <strong>{a:g}</strong> | "
                    f"Window includes unit: <strong>{'yes' if include_self else 'no'}</strong> | "
                    f"Weights: <strong>{html.escape(weight_label)}</strong> (binary) | Multiple testing: <strong>{html.escape(correction)}</strong>")
        body = kpis + a_note + f"<section><h2>Typology</h2>{donut}{scatter}</section>" + guidance
        return report_page(self.DISPLAY, subtitle, body)
