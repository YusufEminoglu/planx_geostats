# PlanX GeoStats Lab — Enhancement Plan v4.0

**Audited:** 2026-09-18 · **Plugin state:** v3.8.1, branch `master`, clean tree
**Scope:** the plugin **and** its reference manual as one deliverable.
**Supersedes:** `ENHANCEMENT_PLAN_v3.9.md` — its two algorithms become Release 1 here,
extended with the anisotropy diagnostics it deferred as shortcoming S1. The v3.9 file is
kept for its `REPORT_v3.9.md` delivery contract, which Release 1 still honours.

---

## 0. Baseline (measured, not assumed)

| Fact | Value | How measured |
|---|---|---|
| Algorithms | 81 | `smoke_provider_catalog.MIN_EXPECTED_ALGORITHM_COUNT`, 81 `addAlgorithm` calls |
| Groups | 6 (`00`–`06`) | `geostats_dock.GEOSTATS_GROUPS` |
| Python LOC | 38,726 | `find -name "*.py" \| xargs wc -l` |
| Manual | 1,034,898 B, 81 `alg-*` cards | `docs/GEOSTATS_REFERENCE_MANUAL.html` |
| Help-link ↔ manual anchor | 81 ↔ 81, no gaps | cross-check of `HELP_SLUGS` against manual `id=` |
| QGIS runtimes | 3.44.12 LTR, 4.2.0 | `Qgis.version()` on both `.bat` launchers |
| QGIS Python | 3.12.13, **identical package set on both runtimes** | direct probe |
| Runtime algorithm matrix | **81/81 PASS on both runtimes** (exit 0, `failed_count: 0`) | `tests/qgis_runtime_algorithm_matrix.py` run directly under each launcher — see the D10 note, `pf.py`'s displayed line is not this number |

Already present in **both** runtimes, so usable without a new dependency:
`numpy 2.4.6`, `scipy 1.17.1`, `pandas 3.0.3`, `scikit-learn 1.9.0`,
`libpysal 4.15.0`, `esda 2.10.0`, `spreg 1.9.1`, `mgwr 2.2.1`, `matplotlib 3.10.9`,
`osgeo.gdal 3.13.1`.
Absent: `spopt`, `giddy`, `pykrige`, `skgstat`, `gstools`.

Two consequences that shape this plan:

1. **`scipy` is available** (it arrives with scikit-learn). v3.9's open question S3 is
   answered: variogram fitting *may* use `scipy.optimize`. But `core/`'s existing engine
   convention is numpy-only — `stats_engines.py` hand-rolls a chi-square survival function
   rather than import scipy — so Release 1 keeps the fit numpy-only for consistency and
   treats scipy as an optional accelerator. See S3 in §5.
2. **Raster output is unproven in this plugin** — every current output is a vector layer
   or an HTML file. Verified working before planning around it: a GDAL Float32 GeoTIFF
   written from numpy reads back through `QgsRasterLayer` as valid, and
   `QgsProcessingParameterRasterDestination` is importable on both runtimes.

---

## 1. Defects found during the audit — fix before new features

These are live, user-facing, and cheap. Fixing them is Release 0.

### D1 — Documentation help link is dead for all 81 algorithms (severity: high)

`algorithms/_mixins.py::DOC_BASE_URL` is
`https://yusufeminoglu.github.io/planx_geostats/GEOSTATS_REFERENCE_MANUAL.html`.
Every algorithm's `helpUrl()` — the "Help" button in the Processing dialog and the dock's
context menu — resolves against it.

```
https://yusufeminoglu.github.io/planx_geostats/GEOSTATS_REFERENCE_MANUAL.html  -> HTTP 404
https://yusufeminoglu.github.io/planx_geostats/                                -> HTTP 404
https://yusufeminoglu.github.io/                                                -> HTTP 404
https://geophilo.com/planx_geostats/GEOSTATS_REFERENCE_MANUAL.html              -> HTTP 200
```

The entire `yusufeminoglu.github.io` host is gone (GitHub account flagged — the same cause
as commit `836a940`). One constant, 81 broken tools.

**Fix:** repoint `DOC_BASE_URL` at the host that actually serves the manual. Then add a
test that pins the constant to a single canonical host so it cannot drift again, and record
the host choice in one place shared by `metadata.txt`, `_mixins.py` and the manual itself.

### D2 — A manual card's body is never closed (severity: medium)

Per-card `<div>` open/close counts, computed by splitting the file on real
`<article …>…</article>` pairs:

| card | `<div>` open | close | |
|---|---|---|---|
| `alg-spatial-durbin` | 12 | 11 | missing one `</div>` |

Exactly one card of 81. The card body opened at line 5616 is never closed, so `see-also` and
Literature sit *inside* the Interpretation Guide box rather than beside it, and the unclosed
element swallows the opening layout of the Methodological Notes that follow the last card.

Browsers recover, so this hides — which is exactly why a tag-balance assertion belongs in the
test suite rather than in a human's eyeballs.

> **Correction to this plan's first draft.** I originally reported two broken cards,
> claiming `alg-gmm` had an extra `</div>` that was closing the SDM card. That was wrong: it
> was an artifact of splitting the file at each `<article>` instead of at its matching
> `</article>`, which swept the Methodological Notes appendix into the last card's count.
> Split correctly, `alg-gmm` is balanced and only `alg-spatial-durbin` is not. The
> document-level count (1158 open / 1157 close) was right and matches the single missing
> close.

### D3 — Raw `<` inside LaTeX corrupts three formulas (severity: medium)

Three occurrences across two lines — 7857 and 7858, in the CatBoost regression entry — where
`\sigma_{<i}` appears in one inline `\(...\)` and twice in one display `$$...$$` block. The
HTML parser reads `<i}` / `<i)` as the start of an `<i>` element, so MathJax never receives
the source string and the formula renders wrong.

The precise predicate matters, and getting it wrong cost me a false-positive sweep: a `<` only
starts a tag when followed by an ASCII letter, `/`, `!` or `?`. The other 31 `<` characters
inside math in this manual — `\(p < 0.05\)`, `\(N < 30\)`, `\(\mathbb{1}[r < 0]\)` — are
followed by a space and are correctly literal text. A test that flags them would be noise; the
shipped test is therefore data-driven (flag a `<letter` only when the name is not a tag the
document actually uses), which finds exactly the 3 real occurrences and nothing else.

### D4 — Manual version string is stale

The manual's `<title>` and its badges read **v3.8.0**; `metadata.txt` is **3.8.1**. Four
occurrences. Cosmetic, but it is the first thing a reviewer checks, and it will be wrong
again after every release until it is derived rather than typed.

### D5 — The published manual is a leftover from the retired GitHub Pages site (severity: medium)

`https://geophilo.com/planx_geostats/GEOSTATS_REFERENCE_MANUAL.html` resolves (308 → 200) but
serves a manual that differs from `docs/GEOSTATS_REFERENCE_MANUAL.html` — the footer credits
"Yusuf Eminoğlu with feedback from educational workflows at **GEOPHILO**" where the repo says
"**Dokuz Eylül University, Department of City and Regional Planning**".

I first filed this as an open branding question for the maintainer. It is not one — three
independent facts settle it:

1. `git log -S "GEOPHILO Spatial Software Studio"` against the repo manual returns **nothing**.
   That wording has never existed in this repository.
2. The repo's `docs/` copy has carried the university wording since `d43bd46`, the commit that
   first added the standalone manual to `docs/`.
3. **The live copy still contains all 3 of D3's `\sigma_{<i}` bugs.** It is an unpatched
   pre-migration snapshot.

So the live artifact came from the old `yusufeminoglu.github.io` host — the same host D1's dead
`DOC_BASE_URL` pointed at — and was carried onto the new domain without being rebuilt from
`docs/`. `metadata.txt` (Hub-validated and authoritative), `README.md` and `CITATION.cff` (author
email `@deu.edu.tr`) all agree with the repo. **The repo is canonical; the deployed copy is
stale.** Refreshing it is a publish, so it belongs to the maintainer (see §6, Q2).


### D6 — The manual exists twice in the repo

Root `GEOSTATS_REFERENCE_MANUAL.html` and `docs/GEOSTATS_REFERENCE_MANUAL.html` are
byte-identical today. They have already drifted once — commit `3b04407`, *"Sync stale
root-level manual duplicate with fixed docs/ copy"*. Two identical copies with no build
step between them is a drift generator. `docs/index.html` is a 323-byte redirect to the
`docs/` copy, so `docs/` is the served location.

### D7 — Four different hosts claim to be this plugin's home; two of them are dead (severity: medium)

Every URL below was fetched, not assumed:

| consumer | URL | status |
|---|---|---|
| `metadata.txt` `repository` / `tracker` | `gitlab.com/geophilo1/planx_geostats` | **200** |
| `metadata.txt` `homepage`, `about` manual link | `geophilo.com/planx_geostats/` | **200** |
| manual footer "Source repository" (GitHub) | `github.com/YusufEminoglu/planx_geostats` | **404** |
| manual footer "Source repository" (GitLab) | `gitlab.com/yusufeminoglu/planx_geostats` | 200 *via redirect* to `geophilo1` |
| `_mixins.py` `DOC_BASE_URL` | `yusufeminoglu.github.io/…` | **404** |
| `CITATION.cff` `url:` | `github.com/YusufEminoglu/planx_geostats` | **404** |

`metadata.txt` is right, and the Hub validates it on upload. The manual footer now points at the
one live repository (fixed in R0, guarded by
`test_manual_source_link_matches_metadata_repository`). The GitHub account is gone — the same
cause as the dead `github.io` site.

### D9 — `CITATION.cff` is four minor versions stale and carries a dead URL (severity: low)

It reads `version: "2.0.0"` and `url: "https://github.com/YusufEminoglu/planx_geostats"` — a 404 —
while the plugin is at 3.8.2.

**Deliberately not fixed here.** `CITATION.cff` is a *generated* artifact, and it is generated in
two steps: `packaging/generate_citations.py` writes the file from `metadata.txt` (which would set
the version and URL correctly), but it also writes `# doi: ""` — **commenting out** the Zenodo
DOI — and a different author email. The real DOI is applied afterwards by
`packaging/apply_dois_and_rebuild_zips.py`. Running only the first script would silently destroy
the DOI `10.5281/zenodo.20753140`. Updating this file correctly means running the maintainer's
release sequence, so it stays as a flagged finding rather than a hand-edit.


### D8 — The smoke suite has been failing since v3.8.1, and hiding 20 further checks (severity: high)

Found only by running it, which is the point:

```
$ py -3 tests/smoke_provider_catalog.py
  File "tests/smoke_provider_catalog.py", line 282, in test_release_documentation_version_is_synchronized
AssertionError: README release verification command should match metadata version
```

The v3.8.1 release bumped `metadata.txt` to 3.8.1 but left README's release-verification
command at `--version 3.8.0`, and `test_release_documentation_version_is_synchronized` asserts
the two agree. Worse than the one failed assertion: `run_all()` executes tests sequentially and
aborts on the first failure, so this test — the **9th of 29** — had been silently preventing the
remaining 20 from running at all. A green-looking suite that stopped executing two thirds of
itself is a stronger warning than a red one.

This is the same class as `docs/TRAPS.md` 4.4 (a suite that prints FAIL but exits 0) and 5.2 (an
assertion satisfied by total failure), except inverted: the suite exits non-zero, so it is
noticed, but the *coverage loss* underneath it is invisible. Fixing the README line restores all
29 checks, which then pass.

---

### D10 — `pf.py` reports the wrong summary line for every QGIS suite (severity: low, cosmetic)

Every `pf.py verify` run on this plugin prints this, for both runtimes:

```
[PASS] qgis[ltr]:   tests/qgis_runtime_algorithm_matrix.py — [WinError 2] The system cannot find the file specified
[PASS] qgis[qgis4]: tests/qgis_runtime_algorithm_matrix.py — [WinError 2] The system cannot find the file specified
```

Both suites in fact pass. Run directly, each prints its real verdict as the **final stdout line**:

```
GEOSTATS_RUNTIME_MATRIX: PASS (81/81)      # qgis_version 3.44.12-Solothurn
GEOSTATS_RUNTIME_MATRIX: PASS (81/81)      # qgis_version 4.2.0-Belém do Pará
```

with `case_count: 81, ok_count: 81, failed_count: 0` in the JSON line ahead of it. The `[WinError 2]`
is benign third-party noise — joblib/loky cannot determine the physical core count and falls back
to logical cores — and it appears at line 378 of 597, nowhere near the summary.

**Mechanism.** `run()` returns `proc.stdout + proc.stderr` — the entire stderr stream appended
after the entire stdout stream. So every warning, including the loky one, lands *after* the real
summary. `result_line()` then walks the lines backwards looking for one containing `passed`,
`PASS`, `FAIL`, `Traceback` or `Error`; it prefers a line containing lowercase `passed`, which
`PASS (81/81)` does not match (case mismatch), so the preference never fires and it falls back to
`interesting[-1]` — the last match, which by then is the loky warning.

**Why this is only cosmetic.** The verdict is computed from `code == 0 and not failed_text and not
silent`, not from the displayed line. The suite returns 1 from `main()` on failure, so a real
failure is caught by the exit code. The cost is a maintainer reading `[WinError 2]` next to a PASS
and having no way to tell whether 81 algorithms ran.

**Fix (one line, shared harness — not applied here).** Make the preference case-correct and prefer
an explicit verdict line:

```python
for line in reversed(interesting):
    if re.search(r"\b(?:PASS|FAIL)\b|--- .* passed|checks passed", line):
        return line
return interesting[-1] if interesting else ""
```

Not applied unilaterally: `packaging/pf.py` is the release gate for all 24 plugins, and this
changes what every plugin's report displays.

### D11 — `failed_text` can read a 10-failure run as zero failures (severity: medium, latent)

In `_run_qgis_suites`:

```python
failed_text = "FAIL" in out and "0 fail" not in out.lower()
```

The intent of the second clause is "ignore a line that says `0 failed`". But it is a **substring**
test, and `0 fail` occurs inside `10 failed`, `20 failed`, … `80 failed`. Measured:

```
failed_count=  9 -> failed_text=True
failed_count= 10 -> failed_text=False   <-- reads as no failure
failed_count= 20 -> failed_text=False
failed_count= 80 -> failed_text=False
failed_count= 81 -> failed_text=True
```

**Not reachable through this plugin today**, because `qgis_runtime_algorithm_matrix.py` returns 1
from `main()` on failure and `sys.exit(main())` propagates it, so `code == 0` already fails. It
matters for exactly the suite shape the surrounding comment describes — one that prints `FAIL` but
exits 0, because it calls `run_all()` at import time. For such a suite, 10/20/30/…/80 failures
would be reported as a PASS. `docs/TRAPS.md` 5.2 is the same family: a predicate satisfied by the
wrong input.

**Fix (not applied here, same shared-harness reason as D10)** — anchor the exemption to a
zero-count verdict line instead of a bare substring:

```python
failed_text = bool(re.search(r"\bFAIL\b", out)) and not re.search(r"\b0\s+failed?\b", out, re.I)
```

---

## 2. Capability gap analysis

Benchmarked against ArcGIS Pro's **Spatial Statistics** toolbox (incl. the 3.7 **Spatial
Component Analysis** toolset), **Geostatistical Analyst**, GeoDa/GeoDaSpace, and the PySAL
stack this plugin already depends on. The PySAL column is not a wish list — it is what
`esda 2.10` and `spreg 1.9.1`, already installed in both runtimes, expose today.

### Group A — Geostatistics: zero coverage (the name gap)

This plugin is called **GeoStats** Lab. It contains no geostatistics: a direct grep for
`kriging`/`variogram` across every algorithm file returns nothing.

| Capability | ArcGIS Pro | GeoDa / PySAL | Plugin today |
|---|---|---|---|
| Empirical semivariogram + model fit | ✅ Geostatistical Wizard | R `gstat`, `scikit-gstat` | ❌ |
| Ordinary / simple / universal kriging | ✅ | `PyKrige`, `gstools` | ❌ |
| Kriging variance as native uncertainty | ✅ | ✅ | ❌ |
| Kriging cross-validation (ME/RMSE/MSDR) | ✅ | ✅ | ❌ |
| Directional variogram / anisotropy | ✅ | ✅ | ❌ |
| Empirical Bayesian kriging (EBK) | ✅ flagship | `PyKrige` EBK | ❌ |
| Indicator kriging (threshold exceedance) | ✅ | ✅ | ❌ |
| IDW / RBF / natural neighbour / trend | ✅ | ✅ | ❌ |
| Conditional Gaussian simulation | ✅ | ✅ | ❌ |

The plugin's existing uncertainty tools (`alg_prediction_uncertainty_map.py`,
`alg_conformal_prediction.py`) are ML-side and resample-based. Kriging variance is
*analytic* — the manual must say so plainly, or a reader will assume they all work alike.

### Group B — LISA family: holes that `esda` already fills

The plugin's flagship group. Against `esda 2.10`'s own inventory:

| `esda` object | Plugin | Why it matters for planning |
|---|---|---|
| `Moran_Rate`, `Moran_Local_Rate`, `assuncao_rate` | ❌ | **Empirical-Bayes rate smoothing.** Planners analyse rates (incidents per 1 000, deaths per 1 000, unemployment share). In small-population units the raw rate is dominated by denominator noise, so Local Moran on raw rates manufactures hot spots. Assunção–Reis is the standard correction and GeoDa ships it. |
| `Geary_Local_MV` | ❌ | Multivariate local Geary — LISA across three or more fields at once. Plugin has univariate Local Geary + bivariate LISA only. |
| `Join_Counts_Local`, `_BV`, `_MV` | ❌ | Local join counts; plugin has the global statistic only. |
| `LOSH` | ❌ | Dedicated local spatial-heterogeneity / outlier statistic (Ord & Getis 2001). |
| `Spatial_Pearson`, `Spatial_Pearson_Local` | ❌ | Spatial correlation between two fields, with inference. |
| `MoranLocalConditional`, `MoranLocalPartial` | ❌ | GeoDa's conditional maps and partial Moran — "does the X–Y relationship hold once Z is controlled?" |
| `Gamma` | ❌ | Generalised cross-product statistic, the family umbrella over Moran/Geary/Getis. |
| `Smaup` | ❌ | Spatial multiscale attribute-uniformity profile. |
| `fdr` | ✅ (v3.8.0) | |
| `Moran`, `Moran_Local`, `Moran_BV`, `Moran_Local_BV`, `Geary`, `Geary_Local`, `G`, `G_Local`, `Join_Counts`, `correlogram` | ✅ | |

### Group C — Regionalization quality: ship the ruler, not just the cutter

The plugin ships SKATER, Multivariate Clustering, DBSCAN/HDBSCAN/GMM — and no way to judge
whether any resulting partition is good. `esda` exposes the metrics:

`boundary_silhouette`, `path_silhouette`, `completeness`, `homogeneity`, `isolation`,
`prominence`, `areal_entropy`, `overlay_entropy`, `external_entropy`.

All pure-`esda`, no new dependency. A regionalization tool without these is a tool whose
output cannot be defended in a review.

### Group D — Regionalization methods

`spopt` (absent; would be a new **optional** dependency, installed through the existing
`00 | Setup and Diagnostics > Install / Update GeoStats Libraries` workflow):

Max-p-regions, AZP, REDCAP, `WardSpatial`, `RegionKMeansHeuristics`. The plugin has SKATER
only. **Max-p is the planning workhorse** — "what is the smallest number of service
districts such that each meets a population floor?" — and it cannot be expressed with
SKATER.

### Group E — Regression: holes `spreg 1.9.1` already fills

Present: OLS, GLR, SAR (`ML_Lag`), SEM (`ML_Error`), SDM, ESF, Regime (`OLS_Regimes`),
Quantile, GWR, MGWR, LM diagnostics, Exploratory.

| `spreg` object | Plugin | Note |
|---|---|---|
| `Probit` | ❌ | Binary outcomes — land-use change, participation, adoption. |
| `PooledOLS`, `ML_LagFE`, `ML_ErrorFE`, `ML_LagRE`, `ML_ErrorRE` | ❌ | **Spatial panel models.** No panel capability at all today. |
| `SUR`, `SURerrorML`, `SURerrorGM`, `SURlagIV` | ❌ | Multi-equation systems with cross-equation correlation. |
| `GM_Error_Het`, `GM_Error_Hom`, `GM_Lag`, `GM_Combo`, `GM_KKP` | ❌ | Heteroskedasticity-robust estimators; `GM_Combo` is the standard endogeneity fallback. |
| `ML_Lag_Regimes`, `ML_Error_Regimes`, `GM_Combo_Regimes` | ❌ | Plugin has regimes for OLS only, not for lag/error. |
| `TSLS_Regimes`, `OLS_Endog_Regimes` | ❌ | Endogenous regressors under regimes. |
| SLX (`slx_lags` argument) | ❌ | SDM is present, plain SLX is not. |

### Group F — Sampling and space–time

- **Spatially balanced sampling** (ArcGIS *Create Spatially Balanced Points*; GRTS / Halton
  / stratified / systematic). Pure numpy is sufficient. Planners need it to design field
  surveys and to pick representative sample points — the plugin has no sampling tool.
- **Space–time** (`giddy`: Markov chains, LISA Markov, spatial dynamics) overlaps the
  separate `planx_datacube` plugin. Cross-reference, do not duplicate.

---

## 3. Releases

Each release is independently shippable, independently verifiable, and ends with the
manual, README, CHANGELOG and metadata in sync.

### R0 — v3.8.2 · Defect release — **BUILT**

D1 · D2 · D3 · D4 · D6 · D7 · D8, plus **seven new guards**, all in
`tests/smoke_provider_catalog.py`:

| guard | what it pins |
|---|---|
| `test_help_url_and_metadata_agree_on_documentation_host` | `DOC_BASE_URL` == `metadata.txt`'s manual link — the D1 failure mode cannot recur |
| `test_help_slugs_match_manual_anchors_one_to_one` | the anchor-id cross-check the `_mixins.py` comment has always *claimed* exists but did not |
| `test_manual_cards_have_balanced_div_tags` | per-card `<div>` balance (D2) |
| `test_manual_has_no_angle_brackets_the_parser_would_read_as_tags` | D3, data-driven so it does not flag `\(p < 0.05\)` |
| `test_manual_version_matches_plugin_version` | D4, all four literals |
| `test_root_and_docs_manual_copies_are_identical` | D6, the drift that already happened twice |
| `test_manual_source_link_matches_metadata_repository` | D7, the manual's "source repository" link (404 today) |

Each was validated against the pre-fix file (`git show HEAD:…`), not just against the fixed
one: the three checks that could be negative-controlled flagged **3**, **1** and **`v3.8.0`**
respectively before the fix and **0**, **0** and **`v3.8.2`** after. A new assertion that
passes against both the broken and the fixed file proves nothing
(`docs/TRAPS.md` 5.2).

**D5 and D9 are not in R0.** D5 needs a redeploy (a publish — the maintainer's call). D9 is a
generated file whose correct update runs the release pipeline (see §1 D9).

No new algorithms. The algorithm count stays 81.

**Evidence:** `%`-in-metadata grep clean · flake8 CI select (E901,E902,E999,F821,F823,F831,
E741,E742,E743) clean · `bandit -q -r . -x ./tests,…` exit 0 · suite 30/30 green, restored from
the 9 that had been running · dual-runtime `pf.py verify` in §4.


### R1 — v3.9.0 · Interpolation and Geostatistics *(new group 07)*

Executes `ENHANCEMENT_PLAN_v3.9.md` and resolves its S1 by *shipping the diagnostic*
rather than deferring it: the fitted model stays isotropic, but the tool shows the
directional empirical variograms that reveal anisotropy when it is there.

1. **Empirical Semivariogram & Model Fitting** — binned empirical semivariogram, nugget /
   sill / range under spherical, exponential, gaussian, linear and stable models, a
   goodness-of-fit measure, **directional variograms + a variogram rose**, and an SVG
   empirical-vs-fitted plot via `core/charts.py`. Emits its fitted parameters in a form the
   kriging tool consumes directly — no retyping numbers between two dialogs.
2. **Ordinary / Universal Kriging Interpolation** — interpolated surface raster **and** a
   kriging-variance raster, driven either by the semivariogram tool's fitted parameters or
   by an inline refit. Leave-one-out cross-validation (mean error, RMSE, mean standardised
   error, RMS standardised error) reported as real numbers in the HTML report.
3. Manual entries for both, registered in the sidebar, `HELP_SLUGS`, provider, dock group
   list and the runtime matrix; two 64×64 icons.

**Design constraints:** numpy-only core in `core/geostat_engines.py`, matching
`stats_engines.py`; no PyKrige-class dependency; raster written through GDAL because
`QgsProcessingParameterRasterDestination` + a GeoTIFF round-trip is verified working;
a stated point-count warning threshold for the O(n³) solve, matching the existing
complexity-warning pattern; any geometry accepted, centroids used, matching every other
tool in the plugin.

### R2 — v3.10.0 · Interpolation completion

IDW · radial basis function (thin-plate, multiquadric) · natural neighbour (Delaunay) ·
global/local polynomial trend · **indicator kriging** (threshold-exceedance probability
surface) · **Empirical Bayesian Kriging** (automated, no manual variogram fitting — the
method a non-specialist planner should actually reach for first) · an interpolation
model-comparison tool reporting cross-validation statistics side by side, mirroring
ArcGIS's *Compare Geostatistical Layers*.

### R3 — v3.11.0 · LISA completion

Empirical-Bayes rate smoothing and EB-adjusted LISA · multivariate local Geary · local
join counts (univariate/bivariate/multivariate) · LOSH · spatial Pearson · conditional and
partial Moran · Gamma · SMAUP. Pure `esda`; no new dependency. The rate-smoothing tool is
the highest-value item in this release and should ship first within it.

### R4 — v3.12.0 · Regionalization

Max-p-regions · AZP · REDCAP, plus the Group C quality metrics wired into every
regionalization and clustering tool. Introduces `spopt` as a new **optional** dependency
through the existing installer workflow.

### R5 — v3.13.0 · Regression completion

Spatial probit · spatial panel (pooled / FE / RE) · SUR family · robust GM family · lag and
error regimes · SLX · endogenous-regressor regimes. All `spreg`; no new dependency.

### R6 — v3.14.0 · Sampling and design

Spatially balanced sampling (GRTS / Halton), stratified and systematic sampling, sample-size
guidance. Pure numpy. Cross-reference `planx_datacube` for the space–time surface rather
than reimplementing it.

### R7 — v4.0.0 · Manual and UX overhaul

Treats the manual as a product, not an appendix:

- **Single source of truth** — one manual file, a build step that derives the served copy
  and the root copy, so D5/D6 cannot recur.
- **A "choosing a spatial weights matrix" chapter** — contiguity vs. KNN vs. distance band
  vs. kernel is the single most consequential choice in this plugin and the manual has no
  chapter on it.
- **A tool-selection decision tree** — the Workflow Advisor exists in code; the manual
  should carry the same guidance as a readable tree.
- **Per-group introductions** and a See-Also graph between related statistics.
- **Derived version strings** (D4) and a print stylesheet.
- **Live-browser QA** — the design-polish lesson from the handoff notes: after any DOM-shape
  change, assert against the real file in a real browser, not just a tag-balance check.

---

## 4. Verification ritual (every phase, both runtimes)

Per `qgis_plugins/AGENTS.md` and `docs/TRAPS.md`, and the ritual already proven on this
plugin's earlier overhaul:

1. `python -m compileall` + `flake8` before/after on every touched `.py` (compare by
   message text, not line number).
2. Manual tag balance; anchor-id cross-check (`name()` ↔ `HELP_SLUGS` ↔ manual `id=`,
   both directions, no orphans).
3. `py -3 packaging/pf.py verify planx_geostats` — both QGIS runtimes, all suites, both
   Hub gates. Bandit is the one that blocks: a blocked version number is burned forever,
   so gate **before** upload, never after.
4. **Do not trust the QGIS line `pf.py` prints** (D10). It surfaces whichever stderr warning
   comes last, not the verdict. Read the suite's own numbers instead:

   ```
   QGIS_CUSTOM_CONFIG_PATH=$(mktemp -d) C:/OSGeo4W/bin/python-qgis-ltr.bat tests/qgis_runtime_algorithm_matrix.py \
     | grep -o 'GEOSTATS_RUNTIME_MATRIX: [A-Z]* ([0-9]*/[0-9]*)'
   ```

   Repeat with `python-qgis.bat`. Both must print `PASS (81/81)` — and the baseline count may
   only go **up**. A `PASS` from `pf.py` alone is not evidence.
5. No literal `%` in `metadata.txt`; no AI attribution in any commit.
6. **Ask before pushing or uploading.** Nothing in this plan is released by an agent.

---

## 5. Known-shortcomings register

| # | Shortcoming | Severity | Release |
|---|---|---|---|
| S1 | Anisotropy: directional variograms shown, but the *fitted model* stays isotropic | Medium | R1 |
| S2 | Universal kriging is a stretch goal within R1; if not shipped, no user-facing text may claim it | Medium | R1 |
| S3 | numpy-only variogram fit deviates from `scipy.optimize`'s accuracy; scipy is verified present on both runtimes but is not a declared dependency, so it stays an optional accelerator | Low | R1 |
| S4 | Kriging is O(n³); needs a stated practical point-count warning, not a silent hang | Medium | R1 |
| S5 | Semivariogram → kriging parameter handoff must be a real contract, not "retype the numbers" | Medium | R1 |
| S6 | Raster output would be this plugin's first; needs its own verification on both runtimes | Medium | R1 |
| S7 | `spopt` (R4) and any future optional package enlarge the install surface; each addition needs a graceful missing-dependency message | Low | R4 |
| S8 | Conditional simulation is out of scope for v4.0 — recorded, not silently omitted | Low | R7 |

---

## 6. Decisions this plan does not make for you

| # | Question | Why it is yours |
|---|---|---|
| Q1 | Which releases to execute now | Effort is yours to spend. R0 is built; R1–R7 are scoped but untouched. |
| Q2 | Redeploy the manual so the live site matches the repo (§1 D5) | It is a **publish**. The evidence says the repo is canonical, but pushing to the public domain is outward-facing and yours to trigger. |
| Q3 | New optional dependency `spopt` for R4 | Widens the install surface. |
| Q4 | Whether to commit R0, and under whose name | Settled for you: the repo's Rule 0 requires commits under Yusuf Eminoğlu alone, with no AI trailer of any kind. Nothing has been committed or pushed. |

**Settled during the audit, so no longer open:** which manual is canonical (Q2's old form —
evidence, §1 D5) and the R0 version number. 3.8.2 is a patch because R0 adds no algorithms and
3.9.0 is already reserved by the pre-existing plan for the geostatistics release; inventing a
minor bump here would collide with it.


---

## 7. Required response format (per executed release)

```markdown
# Implementation Report v<X.Y.Z>
## 1. Release status — each numbered item: DONE | PARTIAL, one line each
## 2. Defect outcomes — D1–D7, plus any NEW defects discovered, numbered D8+
## 3. Capability summary — table of new algorithm id | displayName | inputs | outputs
## 4. Validation evidence — real numbers (fitted variogram params, LOO cross-validation
     error), not "it ran without error"
## 5. Test evidence — verbatim: pure suites, QGIS 3.44 LTR, QGIS 4.x
## 6. Gate outputs — bandit, `%`-in-metadata grep, verilmiş verbatim
## 7. Deviations from this plan
## 8. Commit list — hash + message, one per line
```

**Honesty over completeness.** A PARTIAL with truthful evidence is acceptable. A DONE that
the reviewer's re-run contradicts invalidates the delivery.
