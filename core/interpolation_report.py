# -*- coding: utf-8 -*-
"""Shared report scaffolding for the deterministic interpolators.

Inverse distance weighting, radial basis functions and polynomial trend
surfaces are three different methods with one thing in common: none of them
carries a variance model.  Kriging reports a prediction and an analytic
variance, because the variogram gives it one.  These three report a prediction
and nothing else - so the only honest measure of their quality is
cross-validation, and a report that presents their output without a
cross-validation figure is presenting an untested surface.

That shared consequence is what this module holds.  The pieces here are the
ones that would otherwise be copied three times and drift apart: the page
stylesheet, the cross-validation table, the statement of which validation
protocol was affordable, and the convex-hull mask that marks cells the
observations do not span.

Why the mask is not optional in the reporting: a weighted average of observed
values can only ever fall between them, so extrapolating an IDW surface looks
harmless.  A fitted polynomial has no such bound - a cubic continued past the
data will leave the observed range within a few cells and keep going.  Both
tools show the same report section, because a reader deciding how much of the
map to believe needs the same fact either way.
"""
from __future__ import annotations

import html

import numpy as np

from .analysis_diagnostics import numeric_quality_summary  # noqa: F401  (re-export)
from .interpolation_engines import extrapolation_summary
from .reporting import analyst_guidance_css
from .charts import chart_css

#: Page stylesheet shared by the three deterministic interpolation reports.
#: Kept here rather than inline so a change to one tool's look is a change to
#: all three - the reports are meant to read as one family with the kriging
#: report, which uses the same rules.
BASE_CSS = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
           color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }
    .container { max-width: 860px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px;
                 box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }
    header { border-bottom: 2px solid #edf2f7; padding-bottom: 20px; margin-bottom: 25px; }
    h1 { color: #1a202c; margin: 0 0 5px 0; font-size: 1.6rem; }
    .subtitle { color: #718096; margin: 0; font-size: 0.95rem; }
    .interpretation-box { background-color: #f8fafc; border-left: 5px solid __COLOR__;
                          padding: 20px; border-radius: 4px; margin-bottom: 30px; }
    .status-title { font-size: 1.3rem; font-weight: 800; color: __COLOR__; margin: 0 0 10px 0;
                    text-transform: uppercase; letter-spacing: 0.05em; }
    .status-desc { color: #4a5568; font-size: 0.95rem; margin: 0; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 25px; }
    th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #edf2f7; font-size: 0.9rem; }
    th { background-color: #ebf8ff; color: #2b6cb0; font-weight: 700; text-transform: uppercase;
         font-size: 0.75rem; letter-spacing: 0.05em; }
    .metric-name { font-weight: 600; color: #2d3748; }
    .metric-val { font-family: monospace; font-size: 1rem; font-weight: 600; }
    footer { margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 15px; font-size: 0.8rem;
             color: #a0aec0; text-align: center; }
    section { margin: 28px 0; }
    h2 { color: #1a202c; font-size: 1.15rem; margin: 0 0 12px 0; }
    .next-action { background: #f0fff4; border-left: 5px solid #2f855a; padding: 16px 18px; border-radius: 4px; }
    .variance-box { background: #fffaf0; border-left: 5px solid #dd6b20; padding: 16px 18px; border-radius: 4px; }
    .flag-list { margin: 0 0 12px 0; padding-left: 22px; }
    .flag-list li { margin-bottom: 8px; color: #4a5568; }
"""


def page_css(verdict_color: str) -> str:
    """The shared stylesheet with the verdict colour substituted in.

    The verdict colour is the one value that differs between runs of the same
    tool, so it is the one placeholder rather than a second copy of the sheet.
    """
    return BASE_CSS.replace("__COLOR__", verdict_color)


def report_head(title: str, subtitle: str, verdict: str, verdict_text: str,
                verdict_color: str, extra_css: str = "") -> str:
    """The opening of a report: head, stylesheet, header and verdict box.

    Ends mid-document on purpose - the caller appends its own sections and then
    calls :func:`report_foot`.  Splitting here keeps every tool's sections in
    that tool's file, where they can be read against the method they describe.
    """
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{page_css(verdict_color)}{analyst_guidance_css()}{chart_css()}
{extra_css}</style>
</head>
<body>
<div class="container">
    <header>
        <h1>{html.escape(title)}</h1>
        <p class="subtitle">{subtitle}</p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">Validation: {html.escape(verdict)}</h2>
        <p class="status-desc">{html.escape(verdict_text)}</p>
    </div>
"""


def report_foot(caveats: str, guidance: str) -> str:
    """The closing of a report: caveats, guidance and the footer."""
    return f"""
    {caveats}

    {guidance}

    <footer>
        Generated by PlanX GeoStats Lab interpolation engine.
    </footer>
</div>
</body>
</html>
"""


# ── verdict ──────────────────────────────────────────────────────────────────

def validation_verdict(cv, field_std, method_label):
    """Turn a cross-validation result into a verdict, a colour and a sentence.

    The thresholds are on the normalised RMSE rather than the raw one, because
    the raw one is in the field's own units and cannot be judged without
    knowing the field.  A normalised RMSE of 1 means the interpolator does no
    better than guessing the mean everywhere; that is the bar being applied.

    Returns ``(verdict, color, description, next_action)``.
    """
    if cv is None:
        return (
            "Not validated", "#718096",
            "Cross-validation was turned off for this run, so the surface below "
            "has not been tested against the observations it was built from. "
            f"{method_label} carries no variance model, so there is nothing else "
            "here that measures its error.",
            "Re-run with cross-validation switched on. Without it there is "
            "nothing in this report that says whether the surface is any good, "
            "and no other section substitutes for it.",
        )

    nrmse = cv.get("nrmse")
    protocol = cv["protocol"]
    if nrmse is None:
        return (
            "Not comparable", "#718096",
            f"{protocol.capitalize()} cross-validation ran, but the field has no "
            "spread to normalise against, so the error cannot be judged on a "
            "scale comparable between fields.",
            "Check the field's variation before reading anything into the "
            "surface. An interpolator on a nearly constant field is describing "
            "noise.",
        )

    rmse = cv["rmse"]
    base = (
        f"{protocol.capitalize()} cross-validation error is {rmse:.6g} in the "
        f"field's own units, which is {nrmse:.3f} times the field's standard "
        f"deviation of {field_std:.6g}. "
    )
    if nrmse < 0.4:
        return (
            "Fit for the mapped pattern", "#2f855a",
            base + "The interpolator is reproducing a real spatial pattern, not "
            "inventing one: an error well under the field's own spread means "
            "predictions beat the field mean by a wide margin.",
            "Compare against a second method on the same points before "
            "committing to this surface. A good score says the method suits this "
            "field, not that another would not suit it better.",
        )
    if nrmse < 1.0:
        return (
            "Better than the mean, no more", "#dd6b20",
            base + "The interpolator beats predicting the field mean everywhere, "
            "but not by much. Whether that is enough depends on what the map is "
            "for.",
            "Try the other interpolators on the same points - the model "
            "comparison tool reports their cross-validation scores side by side "
            "- and check whether a different search neighbourhood helps.",
        )
    return (
        "Worse than the mean", "#c53030",
        base + "A normalised error at or above 1 means the interpolator does no "
        "better than returning the field mean at every cell. The surface may "
        "still look plausible, which is exactly why this number is reported.",
        "Do not use this surface as evidence of spatial pattern. Check the field "
        "for a trend, a different search neighbourhood, or a stronger spatial "
        "structure than this method assumes; if nothing helps, the observations "
        "may not support interpolation over this area at all.",
    )


def verdict_description(verdict: str) -> str:
    """A one-line restatement of a verdict, for a summary that has no room
    for the full cross-validation sentence."""
    return {
        "Fit for the mapped pattern": (
            "Cross-validation says the surface reproduces a spatial pattern "
            "rather than inventing one."
        ),
        "Better than the mean, no more": (
            "Cross-validation says the surface is better than the field mean, "
            "but only just."
        ),
        "Worse than the mean": (
            "Cross-validation says the surface does no better than the field "
            "mean. Treat it as a drawing, not as evidence."
        ),
        "Not comparable": (
            "Cross-validation ran, but the field has too little spread for the "
            "error to be judged."
        ),
        "Not validated": (
            "No cross-validation was run, so the surface is untested."
        ),
    }.get(verdict, "See the cross-validation section for what this means.")


# ── cross-validation rendering ───────────────────────────────────────────────

def cv_metric_rows(cv) -> str:
    """The metric table body for a cross-validation result."""
    if cv is None:
        return (
            '<tr><td class="metric-name">Cross-validation</td>'
            '<td class="metric-val">not run</td></tr>'
        )
    nrmse = "n/a" if cv.get("nrmse") is None else f"{cv['nrmse']:.4f}"
    rows = [
        ("Validation protocol", cv["protocol"]),
        ("Observations scored", str(cv["n_scored"])),
        ("Mean error (ME)", f"{cv['me']:.6g}"),
        ("Root mean squared error (RMSE)", f"{cv['rmse']:.6g}"),
        ("Mean absolute error (MAE)", f"{cv['mae']:.6g}"),
        ("Largest single error", f"{cv['max_abs_error']:.6g}"),
        ("RMSE / field standard deviation", nrmse),
    ]
    if cv["n_dropped"]:
        rows.append(
            ("Dropped for too few neighbours", str(cv["n_dropped"]))
        )
    return "".join(
        "<tr>"
        f'<td class="metric-name">{html.escape(label)}</td>'
        f'<td class="metric-val">{html.escape(value)}</td>'
        "</tr>"
        for label, value in rows
    )


def cv_protocol_note(cv) -> str:
    """The paragraph explaining which protocol ran and why that matters.

    The leave-one-out versus k-fold distinction is the single most misread
    number in an interpolation report, so the report states which one it used
    and what the choice cost, rather than showing an unexplained RMSE.
    """
    if cv is None:
        return ""
    if cv["protocol_id"] == "loo":
        return (
            "Every observation was held out in turn and predicted from all the "
            "others. Because interpolation here is exact, a residual against "
            "the fitted surface would be zero by construction and would say "
            "nothing; only holding points out tests the method. Leave-one-out "
            "is the strongest test available and the only one whose result does "
            "not depend on a random split."
        )
    return (
        f"The observations were split into {cv['folds']} groups and each group "
        "predicted from the rest. This is a stricter test than leave-one-out: "
        "each model sees only a fraction of the data, so its error is "
        "systematically larger and is not comparable with a leave-one-out "
        "figure from another run. It was used because leave-one-out is not "
        "affordable at this observation count - it refits once per observation. "
        f"The random split is fixed by a seed, so the {cv['n_scored']} scored "
        "observations are reproducible."
    )


# ── extrapolation mask ───────────────────────────────────────────────────────

def hull_summary(coords, targets):
    """The convex-hull extrapolation summary, or None if the hull is degenerate."""
    return extrapolation_summary(coords, targets)


def apply_hull_mask(grid, hull, masked_value):
    """Replace cells outside the observations' convex hull with the no-data value.

    ``grid`` is the 2-D output surface and ``hull`` the summary dict from
    :func:`hull_summary`, whose ``inside_mask`` is in the same row-major order
    the grid was built from.
    """
    if hull is None:
        return grid
    inside = np.asarray(hull["inside_mask"], dtype=bool).reshape(grid.shape)
    out = np.array(grid, dtype=float, copy=True)
    out[~inside] = masked_value
    return out


def hull_note(hull, masked: bool, nodata) -> str:
    """The sentence stating how much of the output the observations do not span.

    A convex-hull count is deliberately the measure used instead of a distance
    to the nearest observation: a point can sit one metre from an observation
    and still be outside the hull, between two arms of a sampling pattern, and
    that is exactly the case where a polynomial or weighted average is least
    constrained.
    """
    if hull is None:
        return (
            "Every feature shares one coordinate, so there is no convex hull to "
            "test the output against and no mask was applied."
        )
    total = hull["total"]
    outside = hull["outside"]
    fraction = hull["fraction"]
    if total == 0:
        return "The output grid was empty, so no extrapolation test was possible."
    if outside == 0:
        return (
            "Every output cell falls inside the convex hull of the observation "
            "locations, so none of this surface is extrapolation."
        )
    head = (
        f"{outside:,} of {total:,} output cells ({fraction:.1%}) fall outside the "
        f"convex hull of the observation locations - the grid covers a wider "
        f"area than the observations span."
    )
    if masked:
        return (
            head + f" Those cells have been written as no-data "
            f"({nodata:g}) rather than left as numbers, because nothing in the "
            "observations constrains them."
        )
    return (
        head + " They have been left as numbers because the mask was not "
        "requested, but they are extrapolation: read them as the method's "
        "behaviour away from data, not as a description of the field. A "
        "weighted average can only fall between observed values and will look "
        "bland; a fitted polynomial has no such bound and can leave the "
        "observed range entirely."
    )


# ── statistics ───────────────────────────────────────────────────────────────

def surface_stats(values, nodata=None):
    """Min/max/mean/std of a surface, ignoring no-data cells.

    ``nodata`` is the sentinel the caller writes for cells it could not
    compute.  Passing it is not optional bookkeeping: the sentinel is an
    ordinary finite number, so a masked grid summarised without it reports the
    sentinel itself as the surface's minimum and drags the mean towards it,
    while the no-data count reads zero.  That failure is silent - the table
    still fills in, it just fills in wrongly - and it was found by reading a
    generated report rather than by any test, which is the reason this
    parameter exists and the reason every masking caller passes it.

    Leaving ``nodata`` as ``None`` keeps the earlier behaviour of ignoring only
    non-finite cells, which is correct for grids that were never masked.
    """
    arr = np.asarray(values, dtype=float)
    missing = ~np.isfinite(arr)
    if nodata is not None:
        missing |= arr == float(nodata)
    finite = arr[~missing]
    n_nodata = int(missing.sum())
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"),
                "mean": float("nan"), "std": float("nan"),
                "n_finite": 0, "n_nodata": n_nodata}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "n_finite": int(finite.size),
        "n_nodata": n_nodata,
    }


def metric_rows(pairs) -> str:
    """Render ``(label, value)`` pairs as a metric table body."""
    return "".join(
        "<tr>"
        f'<td class="metric-name">{html.escape(str(label))}</td>'
        f'<td class="metric-val">{html.escape(str(value))}</td>'
        "</tr>"
        for label, value in pairs
    )


def fmt(value, digits=6):
    """Format a number for a report table, with a dash for anything missing."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}g}"
