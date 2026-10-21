# -*- coding: utf-8 -*-
"""Shared pieces for the regionalisation tools and the quality report (v3.12.0).

Every tool that cuts a layer into regions - max-p, AZP, REDCAP, SKATER and
Multivariate Clustering - ends with the same question: how good is this
partition?  ``quality_section`` answers it the same way for all of them, so a
SKATER run and an AZP run on the same data can be compared line by line.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from qgis.core import QgsGeometry

from ..core.charts import bar_chart_svg, kpi_card_row_html
from ..core.region_quality import (
    boundary_silhouette,
    homogeneity_completeness,
    isoperimetric_quotient,
    partition_summary,
    path_silhouette,
    silhouette_samples,
)
from ..core.regionalization_engines import adjacency_lists
from ..core.weights import geometry_is_missing_or_empty
from ._local_common import callout, fmt, html_table, report_page

REGION_DOCS = {
    "region_id": "Region (0-based id)",
    "sil": "Attribute silhouette: closer to own region (+1) or to the nearest other region (-1)",
    "bnd_sil": "Boundary silhouette against the regions the unit touches (0 for interior units)",
    "path_sil": "Silhouette on shortest-path distance through the neighbourhood graph",
}


def region_compactness(source, fids, labels):
    """Isoperimetric quotient of each dissolved region (polygon layers only)."""
    by_region: dict[int, list] = {}
    index = {fid: int(lab) for fid, lab in zip(fids, labels) if lab >= 0}
    for feature in source.getFeatures():
        region = index.get(feature.id())
        if region is None:
            continue
        geom = feature.geometry()
        if geometry_is_missing_or_empty(geom):
            continue
        by_region.setdefault(region, []).append(QgsGeometry(geom))
    out = {}
    for region, geoms in by_region.items():
        try:
            merged = QgsGeometry.unaryUnion(geoms)
            area = merged.area()
            out[region] = isoperimetric_quotient(area, merged.length()) if area > 0 else float("nan")
        except (RuntimeError, TypeError, ValueError):
            out[region] = float("nan")
    return out


def quality_section(data, labels, graph, *, compactness=None, reference=None, reference_name="",
                    area_weights=None, extra_region_cols=None, title="How good is this partition?"):
    """Return ``(html, per_unit)`` - the quality report block and the per-unit
    silhouette arrays (``sil``, ``bnd_sil``, ``path_sil``; path is None when
    the layer is too large for all-pairs shortest paths)."""
    labels = np.asarray(labels)
    keep = labels >= 0
    idx = np.flatnonzero(keep)
    x = np.asarray(data, dtype=float)[idx]
    lab = labels[idx]
    if graph is not None:
        adj_full = adjacency_lists(graph)
        remap = {old: new for new, old in enumerate(idx)}
        adj = [[remap[v] for v in adj_full[u] if v in remap] for u in idx]
    else:
        adj = None                     # aspatial partition (K-means): no boundary or path measures
    summary = partition_summary(x, lab, adj, None if area_weights is None else np.asarray(area_weights)[idx])
    sil = silhouette_samples(x, lab)
    if adj is not None:
        bsil = boundary_silhouette(x, lab, adj)
        psil = path_silhouette(x, lab, adj)
        boundary_mask = np.array([any(lab[v] != lab[u] for v in adj[u]) for u in range(len(lab))])
    else:
        bsil = np.zeros(len(lab))
        psil = None
        boundary_mask = np.zeros(len(lab), dtype=bool)
    kpis = kpi_card_row_html([
        {"label": "Regions", "value": str(summary["k"])},
        {"label": "R² (between-region share)", "value": fmt(summary["r2"], 3),
         "tone": "good" if summary["r2"] >= 0.5 else "neutral"},
        {"label": "Mean silhouette", "value": fmt(float(sil.mean()), 3),
         "tone": "good" if sil.mean() > 0.25 else ("warn" if sil.mean() < 0 else "neutral")},
        {"label": "Mean boundary silhouette", "value": fmt(float(bsil[boundary_mask].mean()), 3) if boundary_mask.any() else "n/a",
         "sublabel": f"{int(boundary_mask.sum())} boundary units" if adj is not None else "no neighbourhood graph"},
        {"label": "Mean path silhouette", "value": fmt(float(psil.mean()), 3) if psil is not None else "skipped",
         "sublabel": "" if psil is not None else ("layer too large" if adj is not None else "no neighbourhood graph")},
        {"label": "Size balance", "value": fmt(summary["balance"], 3), "sublabel": "1 = equal-sized regions"},
    ])
    rows = []
    for r in summary["regions"]:
        members = lab == r["region"]
        row = [r["region"], r["size"], r["ssd"], float(sil[members].mean()),
               compactness.get(r["region"], float("nan")) if compactness else float("nan"),
               "n/a" if r["contiguous"] is None else ("yes" if r["contiguous"] else "NO")]
        if extra_region_cols:
            row += [extra_region_cols[name].get(r["region"], "") for name in extra_region_cols]
        rows.append(row)
    headers = ["Region", "Units", "SSD", "Mean silhouette", "Compactness (IPQ)", "Contiguous"]
    if extra_region_cols:
        headers += list(extra_region_cols)
    table = html_table(headers, rows)
    worst = sorted(summary["regions"], key=lambda r: float(sil[lab == r["region"]].mean()))[:min(10, summary["k"])]
    chart = bar_chart_svg([f"region {r['region']}" for r in worst], [float(sil[lab == r["region"]].mean()) for r in worst],
                          title="Regions with the weakest mean silhouette")
    notes = []
    if summary["all_contiguous"] is False:
        notes.append(callout("At least one region is not contiguous on the neighbourhood graph."))
    neg = float((sil < 0).mean())
    if neg > 0.25:
        notes.append(callout(f"{100 * neg:.0f}% of units have a negative silhouette - they resemble another region more "
                             "than their own. For a spatially constrained method some of this is the price of contiguity; "
                             "a large share suggests too many regions or fields that do not separate the area."))
    ref_html = ""
    if reference is not None:
        hc = homogeneity_completeness(lab, np.asarray(reference)[idx],
                                      None if area_weights is None else np.asarray(area_weights)[idx])
        ref_html = ("<h3>Agreement with the reference partition"
                    f"{' (' + reference_name + ')' if reference_name else ''}</h3>"
                    + html_table(["Homogeneity", "Completeness", "V-measure"],
                                 [[hc["homogeneity"], hc["completeness"], hc["v_measure"]]])
                    + "<p>Homogeneity is 1 when every region lies inside a single reference zone; completeness is 1 when "
                      "every reference zone lies inside a single region; the V-measure is their harmonic mean"
                    + (" (area-weighted)." if area_weights is not None else ".") + "</p>")
    html_block = (f"<section><h2>{title}</h2>{kpis}{''.join(notes)}{table}{chart}{ref_html}"
                  "<p class=\"chart-caption\">R² is the share of the (standardised) attribute variance that lies between "
                  "regions. A silhouette near +1 means a unit sits firmly in its region, near 0 on the border between two, "
                  "below 0 closer to another region. The boundary silhouette asks the same only of units on a region edge, "
                  "against the regions they touch; the path silhouette measures distance along the neighbourhood graph, so "
                  "it rewards regions that are both alike and well connected. Compactness is 4πA/P² of the "
                  "dissolved region (1 = circle).</p></section>")
    per_unit = {"sil": np.full(len(labels), np.nan), "bnd_sil": np.full(len(labels), np.nan),
                "path_sil": np.full(len(labels), np.nan)}
    per_unit["sil"][idx] = sil
    per_unit["bnd_sil"][idx] = bsil
    if psil is not None:
        per_unit["path_sil"][idx] = psil
    return html_block, per_unit, summary


def partition_report(path, slug, title, subtitle, section):
    """Write the quality report for a tool that has no other report (SKATER,
    Multivariate Clustering) and return its path."""
    if not path:
        path = os.path.join(tempfile.gettempdir(), f"planx_{slug}_quality.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(report_page(title, subtitle, section))
    return path
