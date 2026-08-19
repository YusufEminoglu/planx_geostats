# GeoStats Lab — Output Visualization Pathway

Status: **planning only** — nothing in this document is implemented yet. This
is the list + outlook the next phase of work will be scoped from.

## 1. Goal

Every one of the 81 algorithms already writes an HTML report
(`_write_html()` in each `algorithms/alg_*.py`), but every one of those
reports is currently text and tables only: numbers, `<table>` rows, and the
shared `analyst_guidance_html()` block. The statistics themselves are often
inherently visual — a Moran scatterplot, a residual plot, a feature-importance
ranking, a confusion matrix — and right now none of that is actually *shown*,
only tabulated.

Next phase: turn the numbers these tools already compute into visuals,
directly inside the same self-contained HTML report each tool already
produces. Not "add charts" narrowly — visualization broadly: proper
statistical diagnostic plots (Moran scatterplots, correlograms, Lorenz
curves, rose diagrams, residual plots) plus a consistent small dashboard
(KPI cards, ranked bars, heatmapped tables) at the top of every report, so a
report is skimmable at a glance and not just a wall of numbers.

## 2. Constraints that shape the technology choice

- **Offline, no new dependency.** The plugin ships to QGIS users who may
  have zero optional packages installed and no internet access at report-open
  time. `matplotlib`/`PIL`-rendered PNGs are out: that is a real dependency,
  a real import cost, and a raster image baked at report-generation time
  (unreadable at odd sizes, not selectable/copyable text, doesn't recolor for
  dark backgrounds).
- **No CDN, no external JS.** `Build-PluginZip.ps1`'s Hub security scan and
  the "self-contained, no network calls" expectation for a QGIS report rule
  out `<script src="https://cdn...">` chart libraries (Chart.js, D3, etc.).
  Anything used must be vendored inline or, better, not be a library at all.
- **The report is opened in QGIS's embedded HTML viewer** (a constrained
  WebEngine view, not a full browser tab), so whatever renders must not
  depend on modern JS framework features or heavy client-side computation.
- **Consistency with the existing pattern.** Every report already calls
  `core/reporting.py::analyst_guidance_html()` / `analyst_guidance_css()` for
  a shared block. The natural place for shared chart helpers is the same
  file, following the same "importable Python function returns an HTML/CSS
  string fragment" shape — no new architecture, no new build step.

**Conclusion: hand-rolled inline SVG, generated as plain Python f-strings,
is the right primitive.** SVG is real HTML-technology markup (not a
rendered image file), needs zero new dependencies, is crisp at any zoom,
inherits page CSS for color/dark-mode, and every value in it is inspectable
in the page source — in keeping with "HTML technologies, not chart/image
generation" as the brief put it. A thin layer of inline (not CDN) vanilla
`<script>` can be added later, purely as progressive enhancement (hover
tooltips), never as a requirement for the chart to render.

## 3. Proposed shared toolkit (`core/reporting.py` additions)

A small set of reusable, dependency-free SVG-fragment builders, each taking
plain Python lists/floats and returning an HTML string, mirroring
`analyst_guidance_html()`'s existing shape:

| Function (strawman name) | Renders | First consumers |
|---|---|---|
| `bar_chart_svg(labels, values, *, horizontal=True, highlight=None)` | Ranked bar chart, optional highlighted bar | Permutation/SHAP/EBM importance, model leaderboards, Q-statistics, join counts |
| `scatter_plot_svg(x, y, *, trend_line=True, quadrants=False, labels=None)` | XY scatter with optional OLS line / quadrant shading | Moran/Lee's L scatterplots, residual-vs-fitted, predicted-vs-actual |
| `line_chart_svg(x, y, *, series=None, reference_line=None)` | One or more line series over a shared x-axis | Correlogram (Incremental Autocorrelation), Ripley's K observed vs expected, partial dependence |
| `histogram_svg(values, *, marker=None, bins=20)` | Binned distribution, optional marked observed value | Sensitivity Test null distribution, prediction-uncertainty spread, GI*/z-score distribution |
| `heatmap_table_svg(matrix, row_labels, col_labels)` | Small grid with cell-intensity shading (reuses `<table>`, adds a `background` scale) | Confusion matrices, correlation/VIF matrices |
| `rose_diagram_svg(angles, *, weights=None)` | Circular/polar histogram | Linear Directional Mean |
| `lorenz_curve_svg(values)` | Cumulative-share curve vs the 45° equality line | Spatial Gini Inequality |
| `kpi_card_row_html(cards)` | A row of 3–5 small stat cards (label, big number, one-line caveat) | The "default dashboard" header — see §4 |

Each returns a self-contained `<svg viewBox=...>…</svg>` (or a small
`<div class="kpi-row">…` block) plus a matching CSS fragment appended to the
existing `analyst_guidance_css()` (or a new sibling `chart_css()` so a report
that doesn't chart anything doesn't pay for unused rules). No `<script>` in
the first pass — every chart must be fully correct as static SVG before any
optional hover-tooltip JS is layered on top.

## 4. The "default dashboard" concept

Every report currently opens straight into prose/tables. Proposal: a small,
consistent header block, right under the `<h1>`, before the detailed
sections — 3 to 5 `kpi_card_row_html()` cards summarizing the single most
important numbers for that tool (e.g., for Global Moran's I: the I value,
the z-score, the p-value; for Random Forest Regression: R², RMSE, OOB score),
followed by that tool's single most important chart. Everything below stays
exactly as it is today (tables, `analyst_guidance_html()`). This gives every
one of the 81 reports the same skimmable shape without redesigning anything
that already works.

## 5. Visualization catalog, by group and priority

**Tier 1 — highest value, and the data is already computed today (no new
statistics needed, purely presentation of existing `results` dict fields):**

- **Global Moran's I / Geary's C / Model Residual Spatial Autocorrelation
  Check** → Moran scatterplot (standardized value vs. spatial lag, OLS trend
  line, HH/HL/LH/LL quadrant shading). *This is the example the brief named
  directly.*
- **Global Bivariate Lee's L / Bivariate Spatial Association** → the same
  scatterplot, bivariate.
- **Incremental Spatial Autocorrelation** → correlogram: line chart of I (or
  z-score) across the already-computed distance increments.
- **Ripley's K-Function** → line chart, observed K(r) vs. CSR-expected K(r)
  across the already-computed distance increments.
- **Spatial Gini Inequality** → Lorenz curve.
- **Attribute Randomization Sensitivity Test (Monte Carlo)** → histogram of
  the simulated null distribution with the observed statistic marked.
- **Permutation Feature Importance / SHAP Global Feature Importance / EBM
  Regression & Classification (mean \|contribution\|)** → ranked horizontal
  bar chart (all three already compute a sorted `(name, value)` list).
- **Partial Dependence Report** → line chart from the existing
  `grid_values`/`average` arrays (currently rendered as a plain table).
- **ML Model Comparison / Model Comparison Matrix (Group 05)** → leaderboard
  bar chart, mean metric with error bars from the existing per-fold std.
- **Spatial k-Fold Cross-Validation Evaluator** → per-fold bar chart (R² or
  accuracy) with the mean as a reference line.
- **Any classifier's confusion matrix** (RF/ET/SVC/GBM×4/MLP/TabPFN/EBM
  Classification) → the existing HTML table becomes a shaded heatmap; same
  underlying `results["confusion_matrix"]`.
- **Conformal Prediction Interval** → sorted-prediction interval plot
  (lower/upper ribbon with the point predictions), a direct visualization of
  `results["lower"]`/`upper"`/`"pred"`.

**Tier 2 — high value, needs a small new aggregation (still no new stats
library, just grouping/binning already-available per-record data):**

- **Getis-Ord Gi\* / Local Moran's I / Bivariate LISA/Lee's L** → cluster-type
  bar or donut breakdown (count of HH/HL/LH/LL/not-significant) alongside the
  existing map output.
- **Colocation Quotient / Join Count Statistics / Geodetector Q-Statistic** →
  bar chart of the per-category or per-strata statistic.
- **Linear Directional Mean** → rose diagram of segment bearings.
- **Regression tools with a fitted/residual pair** (OLS, GLR, SAR, SEM, SDM,
  ESF, GWR, MGWR, every ML regressor) → residual-vs-fitted and/or
  predicted-vs-actual scatterplot from data already written per-record.
- **Prediction Uncertainty Map** → histogram of `unc_std` across records.
- **DiCE Counterfactual Explanation** → small diverging before→after bar per
  changed field for the explained record.

**Tier 3 — valuable but more design work (new layout, not just a new chart
primitive):**

- **SHAP Local Explanation Report** → a proper waterfall chart (base value →
  sequential signed contributions → final prediction), the standard SHAP
  local-explanation visual; needs a dedicated `waterfall_svg()` beyond the
  Tier-1 toolkit.
- **SHAP Spatial Attribution Map** → inherently a map concern (QGIS
  symbology), not an HTML chart; at most a small legend/summary chart, so
  this stays low priority for this pathway specifically.
- **Standard Deviational Ellipse / Directional Distribution** → these already
  write a geometry to the map; an HTML-side mini compass-rose diagram
  showing the rotation/axis ratio would be a "nice to have," not essential
  since the real output is already visual on the map canvas.

## 6. Phased rollout (once we start building)

1. Land the shared toolkit in `core/reporting.py` (§3) with its own smoke
   test (structural: valid SVG produced for representative inputs, matching
   the existing AST/structural-test style already used elsewhere in this
   repo) — no algorithm changes yet.
2. Wire up the Tier 1 list (§5), one tool at a time, each a small, reviewable
   diff: swap a `_write_html()`'s existing table/number for the chart helper,
   verify the runtime matrix case for that tool still passes, spot-check the
   rendered HTML.
3. Add the default-dashboard KPI header (§4) plugin-wide once 2–3 charted
   tools validate the pattern looks right.
4. Tier 2, same one-tool-at-a-time approach.
5. Tier 3 (waterfall chart) as a dedicated small effort, since it needs a new
   primitive rather than reusing the Tier 1 toolkit.
6. Reference manual: add a one-line "Report includes: …" note to each
   updated tool's manual entry so the manual stays in sync with what a user
   actually sees.

## 7. Open questions to settle before implementation starts

- Naming/location: keep everything in `core/reporting.py`, or split chart
  helpers into a new `core/charts.py` imported by `reporting.py` (cleaner
  once the toolkit grows past ~8 functions)?
- Should charts respect a light/dark toggle, or target light-only (matching
  every existing report's fixed `#f6f8fb` background today)?
- Confusion-matrix and heatmap color scale: sequential single-hue (safe,
  colorblind-friendly) vs. diverging — pick once, apply everywhere for
  consistency.
- Where exactly does the KPI dashboard row sit relative to the existing
  `<div class="summary">` block each report already opens with — replace it,
  or sit above it?

---
This file will be superseded by an implementation plan once the list above
is confirmed.
