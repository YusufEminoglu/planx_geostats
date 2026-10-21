# -*- coding: utf-8 -*-
"""Regionalization Quality Assessment Processing algorithm."""
from __future__ import annotations

import html

import numpy as np

from qgis.core import NULL, QgsProcessingParameterBoolean

from ..core.reporting import analyst_guidance_html
from ._local_common import DOUBLE, SpatialStatBase, ToolResult, callout, report_page
from ._region_common import REGION_DOCS, quality_section, region_compactness


def _codes(values):
    """Map arbitrary label values (text or numbers) to 0..k-1; missing -> -1."""
    mapping = {}
    out = []
    for v in values:
        if v is None or v == NULL or str(v) == "NULL" or str(v).strip() == "":
            out.append(-1)
            continue
        key = str(v)
        out.append(mapping.setdefault(key, len(mapping)))
    return np.asarray(out), {code: name for name, code in mapping.items()}


class RegionQualityAlgorithm(SpatialStatBase):
    FIELDS = "FIELDS"
    LABEL_FIELD = "LABEL_FIELD"
    REFERENCE_FIELD = "REFERENCE_FIELD"
    AREA_WEIGHTED = "AREA_WEIGHTED"

    ALG_ID = "regionalization_quality"
    DISPLAY = "Regionalization Quality Assessment"
    GROUP = "03 | Hot Spots and Spatial Outliers"
    GROUP_ID = "planx_hotspots_outliers"
    OUTPUT_LABEL = "Units with quality scores"
    REPORT_LABEL = "Regionalization quality report"
    HELP = (
        "Judges any partition of a layer - the region_id of SKATER, Max-p, AZP or REDCAP, a cluster_id from "
        "K-means or DBSCAN, or an administrative district field - against the attributes it is supposed to "
        "group. A regionalisation tool without these figures is a tool whose output cannot be defended in a "
        "review.\n\n"
        "Measures: R-squared (between-region share of the attribute variance); the attribute silhouette of "
        "every unit (Rousseeuw 1987); the boundary silhouette, which asks the silhouette question only of units "
        "on a region edge and only against the regions they touch; the path silhouette, computed on shortest-"
        "path distance through the neighbourhood graph (Wolf, Knaap & Rey 2021); region size balance; "
        "contiguity of every region on the graph; and the isoperimetric compactness of each dissolved region. "
        "With a reference field (e.g. existing districts) it adds homogeneity, completeness and the V-measure "
        "(Rosenberg & Hirschberg 2007), optionally weighted by unit area.\n\n"
        "Run it on two partitions of the same layer with the same fields to compare them line by line."
    )

    def add_parameters(self):
        self.any_field(self.LABEL_FIELD, "Region / cluster field to assess")
        self.numeric_field(self.FIELDS, "Attribute fields the regions should be homogeneous on", multiple=True)
        self.any_field(self.REFERENCE_FIELD, "Reference partition to compare with (optional)", optional=True)

    def add_trailing_parameters(self):
        self.addParameter(QgsProcessingParameterBoolean(
            self.AREA_WEIGHTED, "Weight the reference comparison by unit area (polygons)", defaultValue=False))

    def compute(self, source, parameters, context, feedback):
        label_field = self.parameterAsString(parameters, self.LABEL_FIELD, context)
        reference_field = self.parameterAsString(parameters, self.REFERENCE_FIELD, context)
        names = [n for n in self.field_list(parameters, self.FIELDS, context) if n != label_field]
        self.require(names, "Select at least one attribute field (other than the region field).")
        area_weighted = self.parameterAsBool(parameters, self.AREA_WEIGHTED, context)
        fids, x, dropped = self.read_matrix(source, names, feedback)
        keep = set(fids)
        labels_raw, refs_raw, areas = {}, {}, {}
        for feature in source.getFeatures():
            if feature.id() in keep:
                labels_raw[feature.id()] = feature.attribute(label_field)
                if reference_field:
                    refs_raw[feature.id()] = feature.attribute(reference_field)
                areas[feature.id()] = feature.geometry().area() if feature.hasGeometry() else 0.0
        labels, names_by_code = _codes([labels_raw[f] for f in fids])
        self.require(len(set(labels[labels >= 0])) >= 2, "The region field needs at least two distinct values.")
        reference = None
        if reference_field:
            reference, _ = _codes([refs_raw[f] for f in fids])
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        weights = np.array([areas[f] for f in fids]) if area_weighted else None
        if area_weighted and not np.any(weights > 0):
            feedback.pushWarning("The layer has no polygon area; the reference comparison is unweighted.")
            weights = None
        section, per_unit, summary = quality_section(
            x, labels, graph, compactness=region_compactness(source, fids, labels),
            reference=reference, reference_name=reference_field, area_weights=weights,
            extra_region_cols={"Value": {code: name for code, name in names_by_code.items()}})
        rows = {fid: {"sil": per_unit["sil"][i], "bnd_sil": per_unit["bnd_sil"][i], "path_sil": per_unit["path_sil"][i]}
                for i, fid in enumerate(fids)}
        guidance = analyst_guidance_html(
            "Regionalization quality",
            "These figures say how well the regions separate the attribute space and whether they respect the "
            "neighbourhood graph; they do not say whether the regions are the right regions for a policy.",
            ["The attribute fields are the ones the regions were meant to be homogeneous on.",
             "Partitions are compared on the same layer, fields and weights.",
             "A spatially constrained partition is judged against another constrained one - an unconstrained K-means "
             "will usually win on silhouette because it is free to ignore contiguity."],
            ["Silhouettes favour compact, convex attribute clusters.",
             "Non-contiguous regions on the chosen graph mean the partition and the graph disagree.",
             "Homogeneity and completeness depend on the number of regions on each side."],
            ["SKATER, Max-p, AZP or REDCAP to produce alternative partitions", "S-maup for the MAUP sensitivity",
             "Multivariate Local Geary for local similarity without a partition"],
            "Report R-squared and the mean silhouette with any zoning you publish, and the weakest regions with it.",
        )
        subtitle = (f"Regions: <strong>{html.escape(label_field)}</strong> | Fields: <strong>{html.escape(', '.join(names))}"
                    f"</strong> | Weights: <strong>{html.escape(weight_label)}</strong>"
                    + (f" | Reference: <strong>{html.escape(reference_field)}</strong>" if reference_field else ""))
        drop = callout(f"{dropped} feature(s) with missing values were left out.") if dropped else ""
        docs = {k: v for k, v in REGION_DOCS.items() if k != "region_id"}
        return ToolResult(report_html=report_page(self.DISPLAY, subtitle, drop + section + guidance),
                          fields=[("sil", DOUBLE), ("bnd_sil", DOUBLE), ("path_sil", DOUBLE)], rows=rows,
                          field_docs=docs, style=("diverging", "sil"),
                          layer_title="PlanX GeoStats regionalization quality")
