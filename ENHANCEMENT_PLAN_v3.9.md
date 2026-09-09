# PlanX GeoStats Lab — Enhancement Plan v3.9.0 (Kriging & Semivariogram Modeling)

**Audience:** an external implementation agent. Work through the phases IN ORDER.
**Repository:** `planx_geostats/` inside the `qgis_plugins` monorepo.
**Current version:** 3.8.0 → **Target version:** 3.9.0 (single release at the end).
**Note:** this is the **first** formal enhancement plan for this plugin — no
prior `ENHANCEMENT_PLAN_*.md` exists to continue from. Baseline verified
fresh against the current tree.

**One paragraph:** This plugin covers spatial autocorrelation (Moran/Geary/
Getis-Ord/LISA/Ripley's K), spatial regression (GWR/MGWR/spatial lag/error/
Durbin/regime), clustering (DBSCAN/HDBSCAN/GMM/SKATER), and a substantial
ML/XAI suite (Random Forest through TabPFN, SHAP, permutation importance,
conformal prediction) — 81 algorithms in total. **It has no kriging and no
semivariogram tool of any kind** — verified by a direct grep across every
algorithm file for "kriging"/"variogram", zero hits. Kriging is the
textbook geostatistical interpolation method the plugin's own name evokes,
and it is structurally different from every existing tool here: it is
neither an autocorrelation *statistic* (Moran's I family), nor a
*regression* (GWR family), nor a generic ML *predictor* (Random Forest
family) — it is the one classical geostatistics workflow (fit a
semivariogram, krige from it, report kriging variance as the native
uncertainty measure) that this "GeoStats Lab" does not yet offer.

---

## 0. Non-negotiable ground rules (read before touching anything)

1. **Read the monorepo docs first:** `qgis_plugins/AGENTS.md` and
   `qgis_plugins/RELEASING.md`. LICENSE present, `experimental=False`,
   release via `release.ps1` are mandatory conventions.
2. **No literal `%` in `metadata.txt`** — `grep -n "%" metadata.txt` must
   return nothing before you finish.
3. **No self-praise wording** anywhere in user-facing copy: never "elite",
   "best", "ultimate", "world-class."
4. **No AI attribution:** never add `Co-Authored-By:` (or any AI) trailers
   to commits.
5. **Never weaken an existing test or gate.** The current algorithm count
   (81) may only go up (to 83, once both new algorithms land). If a test
   blocks you, fix the code, not the test.
6. **`numpy` is already a hard, direct import** across this plugin's
   algorithms (e.g. `algorithms/alg_gwr.py`) — not an optional/on-demand
   dependency like the ML libraries behind `alg_install_libraries.py`. The
   two new algorithms in this plan may rely on `numpy` directly on that
   same basis. Do **not** add a new *required* dependency beyond it; if
   nonlinear semivariogram-model curve fitting genuinely needs `scipy.
   optimize` for a good fit (a plain `numpy.linalg.lstsq` cannot fit a
   nonlinear spherical/exponential/gaussian model), check whether `scipy`
   is already an effective dependency here (it underlies `scikit-learn`,
   which this plugin already treats as available/installable) before
   deciding whether it needs the same on-demand-install treatment as the
   GBM libraries — investigate, don't assume either way.
7. **Reuse existing core infrastructure rather than duplicating it:**
   `core/weights.py` (`geometry_centroid_point`, spatial-weights
   construction), `core/charts.py` (`scatter_plot_svg`, `chart_css`) for the
   empirical-vs-fitted variogram plot, `core/layer_metadata.py`
   (`apply_output_metadata`) for consistent output styling, and the
   `HelpUrlMixin`/icon (`_icons.py::algorithm_icon`) conventions every
   existing algorithm file already follows.
8. **Security scan:** run `bandit -r . -x tests,__pycache__` before
   finishing.
9. **Do not bump the version or write the changelog until Phase 3. Do not
   release/tag/upload** — the maintainer releases manually after review.
10. One commit per completed phase, e.g.
    `phase-1: add Empirical Semivariogram & Model Fitting algorithm`.

---

## Phase 1 — Empirical Semivariogram & Model Fitting

**What.** A new algorithm, `algorithms/alg_semivariogram.py`
(`name()`: `empirical_semivariogram`, `displayName()`:
`"Empirical Semivariogram & Model Fitting"`). Given a point layer and a
numeric field, computes the empirical semivariogram (binned average
squared half-difference of paired values against pairwise distance, the
standard textbook definition) and fits the user's choice of the four
standard theoretical models — **spherical, exponential, gaussian, linear**
— reporting the fitted **nugget, sill and range** parameters, a goodness-
of-fit measure, and an SVG plot (empirical points + fitted curve, reusing
`core/charts.py`) as an HTML output alongside a parameter table. This
output feeds Phase 2's kriging tool directly — its three fitted parameters
are exactly what ordinary/universal kriging needs to build its covariance
model.

**Design notes.**
- Distance-bin count and max-lag-distance are user parameters with
  sensible defaults (e.g. max lag = a third of the study area's diagonal,
  a common geostatistical rule of thumb — cite it, don't invent a number
  with no justification).
- Anisotropy (direction-dependent variograms) is a real, common extension
  of this technique but is **out of scope for this phase** — ship the
  isotropic (direction-independent) case correctly and completely, and
  record directional/anisotropic variograms as a deferred shortcoming
  (S-register below), not a silent omission.
- Output the fitted model parameters in a form Phase 2 can consume
  directly (e.g. a small structured table layer or a documented dict
  contract) rather than requiring the user to retype numbers between the
  two tools by hand.

---

## Phase 2 — Kriging Interpolation

**What.** A new algorithm, `algorithms/alg_kriging.py`
(`name()`: `kriging_interpolation`, `displayName()`:
`"Ordinary / Universal Kriging Interpolation"`), producing an interpolated
raster surface from a point layer + numeric field + a fitted variogram
model (either re-fit inline using Phase 1's logic, or consuming Phase 1's
output — decide which based on what is actually more usable in the
Processing Toolbox/model-builder context, and document the choice).
**Ordinary kriging** (constant unknown mean) is the required baseline;
**universal kriging** (a polynomial drift term for non-stationary data) is
a stretch goal within this phase, not a separate one — ship ordinary
kriging correctly first, add universal kriging only if it does not put
ordinary kriging's correctness or the test schedule at risk.

Two outputs, matching this plugin's existing pattern of surfacing
uncertainty as a first-class result (see `alg_prediction_uncertainty_map.py`,
`alg_conformal_prediction.py`): the **interpolated surface** raster, and a
**kriging variance** raster (kriging's own native, analytically-derived
uncertainty measure — not a bootstrap or Monte Carlo approximation like
some of the plugin's ML-side uncertainty tools, and the manual must state
this distinction plainly since a reader familiar with the plugin's other
uncertainty tools should not assume they all work the same way).

**Design notes.**
- The kriging linear system (solving for point weights from the fitted
  covariance/semivariogram model at each prediction location) is a
  standard, well-documented matrix solve — implement it with `numpy`
  directly, matching ground rule 6; do not introduce a full external
  geostatistics library (e.g. PyKrige) as a new dependency when the core
  algorithm is a bounded, well-understood linear-algebra problem this
  plugin's existing `numpy` usage already supports.
- Performance: a naive kriging implementation is O(n³) in the number of
  input points for the covariance matrix solve. State the plugin's chosen
  practical point-count guidance in the manual (a warning threshold, not a
  hard cap) rather than letting it hang silently on a large input, matching
  this plugin's existing pattern of point-count/complexity warnings
  elsewhere (e.g. `alg_sensitivity_test.py`, `alg_spatial_cv_evaluator.py`
  — check their actual pattern and match it, don't invent a new one).

---

## Phase 3 — Documentation, registration, version bump

Only after Phases 1–2 are done and tested:
- Register both algorithms in the provider and any dashboard/catalog file
  this plugin uses (check whether `planx_geostats` has the same
  provider-plus-dashboard-catalog dual-registration convention documented
  for `planx_suitability_lab` in that plugin's own enhancement history — if
  so, both places need the new algorithms; if this plugin only has a single
  provider registration, confirm that and don't invent a second file to
  update).
- Write `shortHelpString()`/`helpUrl()` for both, matching this plugin's
  own existing per-algorithm help-text depth and structure (check a
  representative existing entry, e.g. `alg_gwr.py`'s, for the actual bar
  to match).
- Add both to `docs/GEOSTATS_REFERENCE_MANUAL.html` following this
  plugin's established per-algorithm entry structure (this session's Track
  A pass already confirmed the manual's search/scroll-spy correctly covers
  new entries added this way — no doc-tooling changes needed, just content).
- Update `README.md`'s algorithm count (81 → 83) and `CHANGELOG.md` with a
  new `## [3.9.0]` entry.
- Bump `metadata.txt`'s `version=3.9.0`; confirm the `%` gate is clean.
- Do **not** tag, run `release.ps1`, or upload.

---

## Known shortcomings register (fix or explicitly defer with reason)

| # | Shortcoming | Severity | Phase |
|---|---|---|---|
| S1 | Anisotropic/directional variograms are out of scope for this plan — isotropic only; must be stated explicitly in the manual, not silently assumed | Medium | 1 |
| S2 | Universal kriging (polynomial drift) is a stretch goal, not a requirement — if not shipped, the algorithm's `displayName()`/help text must not claim it | Medium | 2 |
| S3 | Whether `scipy` is genuinely available/needed for nonlinear variogram-model fitting must be verified, not assumed either way (ground rule 6) | Medium | 1 |
| S4 | Kriging's O(n³) cost needs a stated practical point-count guidance, matching this plugin's existing complexity-warning pattern elsewhere — verify that pattern before inventing a new one | Medium | 2 |
| S5 | Confirm whether Phase 2 re-fits the variogram inline or consumes Phase 1's output as a hard input; whichever is chosen, the two algorithms' parameter contracts must actually match | Medium | 1, 2 |

---

## R. Required response format

Deliver a single file `REPORT_v3.9.md` in the plugin root:

```markdown
# Implementation Report v3.9.0
## 1. Phase status
Phase 1: DONE|PARTIAL — <one line>
Phase 2: DONE|PARTIAL — <one line, state ordinary-only vs. ordinary+universal>
Phase 3: DONE|PARTIAL — <one line>
## 2. Shortcomings register outcomes
S1–S5, plus any NEW shortcomings discovered, numbered S6+
## 3. New algorithm summary
| algorithm id | displayName | inputs | outputs | model(s) supported |
(2 rows)
## 4. Validation evidence
(a synthetic or sample-data test showing the fitted variogram parameters
are sane, and a kriged surface's cross-validation error — e.g. leave-one-out
— reported as a real number, not just "it ran without error")
## 5. Test evidence
### 5.1 pure test output (verbatim, complete)
### 5.2 e2e QGIS 3.44 LTR output (verbatim, complete)
### 5.3 e2e QGIS 4.x output (verbatim, complete)
## 6. Gate outputs
bandit / `%`-in-metadata grep (verbatim)
## 7. Deviations from this plan
## 8. Commit list
(hash + message, one per line)
```

**Honesty over completeness:** a PARTIAL with truthful evidence is
acceptable; a DONE that the reviewer's re-run contradicts invalidates the
delivery.
