# Implementation Report v3.9.0

Release: R1 — Interpolation and Geostatistics (PlanX GeoStats Lab).
Scope: two new Processing algorithms, one new analytical group, the reference
manual, and the counters/registries that gate them.
Nothing was released, tagged, pushed or uploaded. The maintainer releases
manually after review.

## 1. Phase status

Phase 1: **DONE** — `core/geostat_engines.py` plus the *Empirical Semivariogram
& Model Fitting* algorithm: empirical binning, five models, Cressie-weighted
least-squares fit, directional rose, parameter layer, HTML report.

Phase 2: **DONE** — **ordinary and universal** kriging are both shipped. The
*Ordinary / Universal Kriging Interpolation* algorithm fits internally or reads
the fitted-parameter layer, kriges with `none` / `linear` / `quadratic` drift,
writes prediction and variance GeoTIFFs, and reports leave-one-out validation.
Universal kriging is not ordinary-only.

Phase 3: **DONE** — both algorithms registered in the provider, a new group
`07 | Interpolation and Geostatistics` in both group registries, two 64×64
icons, `HELP_SLUGS` entries, the manual's group-07 section and version/count
strings, `metadata.txt` / `README.md` / `CHANGELOG.md` /
`VISUALIZATION_PATHWAY.md` / `QA_MANUAL_TEST_MATRIX.md`, and the test gates
raised from 81 to 83. All gates re-run green on both QGIS runtimes.

## 2. Shortcomings register outcomes

### S1 — Anisotropy out of scope for this plan (Medium)
**Resolved by scope, stated explicitly.** The *fit* is isotropic and the manual
says so in the semivariogram card: "the fitted model is *isotropic* … The
directional rose is a diagnostic that shows anisotropy when it is present; it is
not absorbed into the fit, and kriging built from these parameters is
isotropic." Directional empirical variograms are computed and displayed
(`N_DIRECTIONS`, `dir_ratio` in the parameter layer) but never enter the fit.
This is a documented limitation, not a silent assumption.

### S2 — Universal kriging was a stretch goal (Medium)
**Resolved — shipped.** `displayName()` is "Ordinary / Universal Kriging
Interpolation"; `DRIFT` selects `None - ordinary kriging`, `Linear trend -
universal kriging` or `Quadratic trend - universal kriging`; the manual's
kriging card derives both systems separately. The claim the help text makes is
the claim the code implements.

### S3 — Is SciPy available/needed for the fit? (Medium)
**Resolved by measurement: not used, not needed.** `grep -n scipy` over
`algorithms/alg_semivariogram.py`, `algorithms/alg_kriging.py` and
`core/geostat_engines.py` returns nothing. The imports are `numpy`, the QGIS
API and the plugin's own core modules. Levenberg–Marquardt is implemented in
`fit_variogram` on top of `numpy` alone, with the parameters carried as
`theta = log([...])` so they stay positive by construction. No new required
dependency was added (ground rule 6); PyKrige, gstools and skstat were not
introduced.

### S4 — Kriging's O(n³) cost needs stated practical guidance (Medium)
**Resolved with measured numbers, in the plugin's existing style.** The
algorithm's `shortHelpString()` carries the measured cost — "200 x 200 surface
took about 1.5 seconds with 200 observations, 18 seconds with 800, and two
minutes at 2000" — and `MAX_POINTS` (default `KRIGING_SOFT_LIMIT = 2000`, hard
range 10–200 000) refuses an oversized run with an actionable message instead of
hanging. `LOO_SOFT_LIMIT = 600` marks leave-one-out as slow rather than silently
running it. The report states the cap the run was bounded by (asserted by the
test harness: "report contains the cap the run was bounded by").

### S5 — Re-fit inline, or consume Phase 1's output? (Medium)
**Resolved: both, through one shared contract.** `PARAM_SOURCE` selects "Fit a
new model from this layer" or "Read the fitted-parameter layer from the
semivariogram tool". Both paths agree because the field list lives in one place,
`core/geostat_engines.PARAM_LAYER_FIELDS`, which the semivariogram tool builds
its output from and the kriging tool validates its input against. The harness
asserts all three properties: the parameter layer's fields equal
`PARAM_LAYER_FIELDS`; the kriging tool's internal fit reproduces the
semivariogram tool's parameters; and the handover run reproduces the same
surface as the internal fit to `max |d| = 2.384e-07`, inside Float32 ulp at that
value (4.8e-07).

### S6 — NEW: universal kriging fitted its variogram to raw trended values
**Found by a failing test, fixed.** `fit_variogram` was being handed the raw
values, so under a trend the variogram rose with distance and the fit answered
with an inflated range instead of the field's. Measured on a synthetic field
with true range 220 map units: fitting the raw trended values gave range
**513** (R²=0.89, which is why it looked healthy), and universal kriging then
lost to ordinary kriging (RMSE 1.48865 vs 1.48006). Fitting the drift residuals
recovered range **238**, and universal kriging won (1.46713 vs 1.48006). Fixed
by `detrend()`: the trend is fitted by least squares before the variogram is
estimated, the variogram describes the residuals, and the report says so — the
harness asserts "UK report discloses the residual fit" and "OK report does not
claim a residual fit". On an untrended field universal kriging is correctly
slightly worse, which is the expected behaviour and not a regression.

### S7 — NEW: the shared caution text misdescribed interpolation tools
**Found by reading the report against itself, fixed.** The plugin-wide
`caveats_html` helper emitted "the distance band or K value" — true for the
pattern-statistics tools it was written for, meaningless for a variogram or a
kriging run. `caveats_html` gained an optional `method_caveats` argument;
passing it replaces those two pattern-statistics lines and the `is_constant` /
neighbourhood appends still apply in both cases, so **the 81 pre-existing
algorithms' reports are byte-identical**. New text for the semivariogram covers
the two things its fit statistics cannot: that the model is fitted to binned
averages and inherits their sampling error (bin count and maximum lag are
analyst choices that move the fit), and that parameters describe the field *as
sampled* — carrying them to another area, density or support is unsupported.

### S8 — NEW: the unchunked right-hand side needed ~18 GB
**Found by measuring, fixed.** Assembling one solve for every target cell
allocates one double per (observation + drift) pair per cell: at 600
observations over a 4-million-cell grid the measured requirement was about
18 GB, which would have failed on any ordinary machine. `krige_chunked` predicts
in chunks with the right-hand side under a fixed memory budget. The harness
proves the refactor is behaviour-preserving: for `none`, `linear` and `quadratic`
drift, one chunk is bit-identical to `krige`, many chunks match to
`max rel diff 0.00e+00`, and the chunk order is preserved.

### S9 — NEW: the range-saturation warning could never fire
**Found by running the tool on the sample data, fixed; negative control run.**
`quality_flags(fit, loo, max_lag)` raises the single most useful warning this
tool has — the fitted range reached the largest distance binned, so the
variogram was still rising at the last bin and the range describes the extent of
the data rather than the field — but the semivariogram algorithm called it as
`quality_flags(fitted)`, without `max_lag`. The flag was therefore dead code in
the shipped tool, even though the engine test covered it. This was caught by
generating a report on the bundled İzmir sample and searching it: the fitted
numbers were present (`39654`, `488.75`, `57.5883`) and the warning text was
absent. Fixed to `quality_flags(fitted, max_lag=lag_limit)`, and the scratch
harness gained three checks that bin a field with true range 220 m only out to
150 m — a fit guaranteed to saturate. **Negative control:** with the fix
reverted, the harness reports `FAIL report carries the still-rising range
warning` (78/79); with the fix in place, `79/79`. The check can fail, so it is
not vacuous.

### S10 — Pre-existing: three whole-plugin flake8 findings (out of scope)
`py -3 -m flake8 --select=F .` reports exactly three, in three files this
release does not touch:

```
.\algorithms\alg_random_forest_regression.py:11:1: F401 'qgis.core.NULL' imported but unused
.\algorithms\alg_spatial_durbin.py:5:1: F401 'math' imported but unused
.\core\advanced_stats_engines.py:844:5: F841 local variable 'p' is assigned to but never used
```

`git status --porcelain` confirms none of the three is in this release's diff.
Recorded rather than silently fixed: repairing unrelated files inside a release
commit would make the diff harder to review.

### S11 — Pre-existing: 38 CRLF warnings from `validate_plugin.py --strict`
All 38 are in `algorithms/`, and 36 of the 38 are files this release never
touches. The cause is environmental, not editorial: `core.autocrlf=true` (and
there is no `.gitattributes`), so git checks files out with CRLF while the
repository blobs are LF — `git show HEAD:algorithms/alg_gwr.py` has 0 CRLF while
the working copy has 703. No file created or edited by this release has mixed
line endings: `algorithms/_mixins.py` is 104 CRLF / 0 bare LF,
`algorithms/alg_semivariogram.py` and `algorithms/alg_kriging.py` are pure LF,
and `git diff` contains zero carriage returns. `algorithms/alg_gwr.py` does have
mixed endings (703 CRLF / 68 bare LF) but was already in that state and is not
in this release's diff.

### S12 — Pre-existing: `QgsField` deprecation warning on QGIS 3.44
On the 3.44 LTR runtime *every* `QgsField(name, QVariant.X)` form emits
`DeprecationWarning: QgsField constructor is deprecated`, which is why the
runtime matrix log is noisy. Probed on both runtimes: on QGIS 4.2 the same calls
are silent and produce the correct field types. The genuine 4.2 hazard is a raw
Python `int` as the second argument (`TypeError: … argument 2 has unexpected
type 'int'`), and `grep -E 'QgsField\([^,]+,\s*[0-9]'` over the plugin returns no
matches — no file passes one. The deprecation is pre-existing and plugin-wide,
cosmetic on 3.44, and silent on 4.2.

### S13 — Pre-existing: one stale test in `packaging/` (out of scope)
`packaging/test_verify_release_zip.py` fails
`test_processing_zip_rejects_ui_hooks_and_missing_algorithm_icons` on its second
assertion, because its synthetic fixture is rooted at `planx` while
`plugins.toml` marks that plugin `ui = true`; `verify_release_zip.py` then
skips the QAction check by design, so the expected error is never produced.
Demonstrated directly: `load_registry().get("planx").ui` is `True`, so
`is_hybrid` is `True`. `packaging/` and `plugins.toml` are not in this release's
diff, and the failure has nothing to do with `planx_geostats`. The verifier
itself passes on a real `planx_geostats` zip (§6).

## 3. New algorithm summary

| algorithm id | displayName | inputs | outputs | model(s) supported |
|---|---|---|---|---|
| `empirical_semivariogram` | Empirical Semivariogram & Model Fitting | `INPUT` (any geometry, centroids), `FIELD` (numeric), `N_BINS` (4–200, default 15), `MAX_LAG` (≥0, 0 = half the largest separation), `MODEL`, `ALPHA` (0.1–2.0), `N_DIRECTIONS` (0–24, default 8), `DIRECTION_TOLERANCE` (5–90°, default 22.5) | `PARAMS_OUT` (single-feature parameter layer: model, nugget, psill, sill, range, alpha, r2, rmse, bins, npoints, dir_ratio), `HTML_REPORT` | spherical, exponential, gaussian, linear, stable (stable uses `ALPHA`; ignored by the other four) |
| `kriging_interpolation` | Ordinary / Universal Kriging Interpolation | `INPUT`, `FIELD`, `PARAM_SOURCE` (fit here / read the parameter layer), `PARAMS_LAYER`, `MODEL`, `ALPHA`, `N_BINS`, `DRIFT` (none / linear / quadratic), `CELL_SIZE` (≥0, 0 = about 200 cells on the longer side), `MAX_POINTS` (10–200000, default 2000), `RUN_LOO` | `PREDICTION` (Float32 GeoTIFF), `VARIANCE` (Float32 GeoTIFF, optional), `HTML_REPORT` | ordinary kriging (`none` drift) and universal kriging (`linear` / `quadratic` drift); the same five variogram models |

Both are registered under `07 | Interpolation and Geostatistics`
(`planx_interpolation_geostatistics`).

## 4. Validation evidence

**Fitted parameters are sane against a known field.** The synthetic field used
by the harness is built with an exponential covariance of practical range
**220 m** and sill **4** on a 0–1000 m box. Fitting the exponential model
recovered

```
sill=3.783 range=219.7      (true 4 and 220)
```

and the Processing output matched a direct engine fit to 1e-6 relative
(`engine sill=3.7827 range=219.727`). The other model families return different
ranges on the same data (spherical 163.7, gaussian 143, stable 150.4, linear
429) as they must: each family has its own practical-range convention, and the
data were generated with an exponential covariance. The reported R² was 0.91 for
the correct family.

**Kriged surface cross-validation is a real number.** On the bundled İzmir FUR
street-network sample (387 usable observations of `road_density`), leave-one-out
over every observation returned

```
ME=-0.193439, RMSE=8.31183, MSE=-0.0105472, RMSSE=0.947981
```

RMSSE ≈ 0.95 against an ideal of 1.0 says the kriging variance is honest about
the error the surface actually produces; ME and MSE near zero say the surface is
not systematically high or low. The run wrote a real 200 × 136 (27 200 cell)
Float32 grid with automatic cell size 216.375 map units.

**Universal kriging earns its keep only where there is a trend.** Measured on
the trended synthetic field: UK RMSE 1.46713 vs OK RMSE 1.48006 when the
variogram is fitted to drift residuals (the shipped behaviour). With the S6
defect in place, the same comparison reversed — UK 1.48865 vs OK 1.48006.

**Stated honestly:** on the İzmir sample the fitted range (39 654) exceeds the
largest distance binned (23 092), so the variogram had not levelled off and the
range there describes the extent of the data rather than a real limit to
similarity. The report now says exactly that (S9) — the delivery rests on the
cross-validation numbers above, not on that range. This is the expected shape of
a city-wide density field with a regional gradient, and drift order is the
diagnostic for it, not a defect in the fit.

## 5. Test evidence

### 5.1 Pure test output (verbatim, complete)

`core/geostat_engines.py` against the validated prototype — `py -3
scratch/verify_engine.py`, exit 0:

```
== pure-numpy import ==
  ok   module imports without QGIS
  ok   MODELS / DRIFTS exposed

== gamma conventions ==
  ok   gamma(0) == 0 even with a nugget
  ok   gamma(0) == 0 for every model
  ok   gamma at large lag == sill
  ok   spherical reaches the sill exactly at h=rng
  ok   unknown model raises

== empirical semivariogram ==
  ok   bins populated -- 15 bins
  ok   N(h) conserved over the binned span -- sum(N(h))=1016 pairs<=max_lag=1016 (of 1770 total, 754 beyond the span)
  ok   default max_lag is half the max separation
  ok   binned mean matches brute force

== directional variogram (pairs, not points) ==
  ok   both cones carry a usable number of pairs -- x=4548 y=502
  ok   cone mask selects a strict subset of its own pair cloud -- cone=6998 of 9730 pairs
  ok   directional bins == hand-binned pairs
  ok   filtering points instead of pairs gives a DIFFERENT answer (so the check above is not vacuous) -- pairs-form sum=4548 point-form sum=7196

== model fitting ==
  ok   spherical: positive parameters -- nug=0.045 psill=5.739 rng=16.200 R2=0.943
  ok   exponential: positive parameters -- nug=0.000 psill=8.883 rng=42.346 R2=0.937
  ok   gaussian: positive parameters -- nug=0.687 psill=5.167 rng=13.578 R2=0.929
  ok   linear: positive parameters -- nug=0.430 psill=6.051 rng=15.611 R2=0.910
  ok   stable: positive parameters -- nug=0.416 psill=5.915 rng=17.686 R2=0.936
  ok   fitted curve tracks the empirical variogram -- R2=0.937
  ok   too few bins raises
  ok   too few bins raises
  ok   flat-at-zero field raises
  ok   flat-at-zero field raises

== kriging ==
  ok   exact at data points when the nugget is 0 -- max|dz|=2.04e-14
  ok   zero variance at data points when the nugget is 0 -- max var=2.09e-15
  ok   nugget=0.5: variance at a REMOVED datum is positive -- var=1.1775
  ok   nugget=1.0: variance at a REMOVED datum is positive -- var=1.7158
  ok   nugget=2.0: variance at a REMOVED datum is positive -- var=2.7094
  ok   SK reaches the fitted sill far away (oracle) -- SK=5.1456 sill=5.1456
  ok   OK far away is at least the sill (sill + mean uncertainty) -- OK=7.1621 sill=5.1456 (+39%)
  ok   OK far away stays within a modest factor of the sill -- OK=7.1621
  ok   a single observation raises
  ok   a single observation raises

== universal kriging ==
  ok   UK beats OK on a trended field -- UK=0.1823 OK=0.1857

== leave-one-out ==
  ok   LOO mean standardised error near 0 -- MSE=-0.000
  ok   LOO RMS standardised error near 1 -- RMSSE=1.496
  ok   LOO reports a real RMSE -- RMSE=1.305

== anisotropy rose ==
  ok   rose returns a range per direction or NaN
  ok   rose finds at least some fittable directions -- 6/6 fitted; ratio=47.55
  ok   rose reports the anisotropy of a stretched field as ratio > 1.5 -- ratio=47.55

== contract + flags ==
  ok   contract carries the five keys
  ok   quality_flags returns a list of strings
  ok   quality_flags fires on a bad fit and dominated nugget -- 4 flags

== fitted curve ==
  ok   fitted_curve starts at gamma=0
  ok   fitted_curve is monotone for a spherical model

== krige_chunked: exact refactor, chunking only shifts the last bits ==
  ok   one chunk is bit-identical to krige (drift=none)
  ok   many chunks match krige to round-off (drift=none) -- max rel diff 0.00e+00
  ok   chunks stay in order: chunked[k] == krige[k] (drift=none)
  ok   one chunk is bit-identical to krige (drift=linear)
  ok   many chunks match krige to round-off (drift=linear) -- max rel diff 0.00e+00
  ok   chunks stay in order: chunked[k] == krige[k] (drift=linear)
  ok   one chunk is bit-identical to krige (drift=quadratic)
  ok   many chunks match krige to round-off (drift=quadratic) -- max rel diff 0.00e+00
  ok   chunks stay in order: chunked[k] == krige[k] (drift=quadratic)
  ok   chunked over zero targets returns empty arrays
  ok   progress is called once per chunk and ends at the total -- 61 calls
  ok   progress returning False cancels -- Kriging cancelled.

59/59 checks passed
ENGINE_OK
```

Line 21 is worth reading twice: the harness first shows that the directional
mask selects a strict subset of its own pair cloud, then shows that the *wrong*
implementation (filtering points instead of pairs) gives a different answer —
so the check above it is not vacuous.

Algorithm-level harnesses, run on both runtimes; the QGIS 3.44 LTR outputs in
full are in §5.2 and §5.3, both exit 0:

| script | QGIS 3.44.12 LTR | QGIS 4.2.0 |
|---|---|---|
| `scratch/run_semivariogram.py` | `79/79 checks passed` — `SEMIVARIOGRAM_OK` | `79/79 checks passed` — `SEMIVARIOGRAM_OK` |
| `scratch/run_kriging.py` | `46/46 checks passed` — `KRIGING_OK` | `46/46 checks passed` — `KRIGING_OK` |

The semivariogram harness checks the shared field contract, algorithm metadata,
all five models end to end (report + parameter layer + round-tripped model
name), the QGIS-vs-engine fit equality, report content, the S9 saturation
warning with its negative control, and error handling. The kriging harness
checks metadata and the exact parameter set, a full ordinary-kriging run
(raster validity, CRS, extent, value range, non-negative and non-constant
variance, grid shape), grid equality against a direct engine call, the
parameter handover, universal kriging on a trended field, report content, the
residual-fit disclosure, the omitted-variance path, polygon input via
centroids, and five error paths.

Because the harnesses create QGIS objects that are still alive at process exit,
`app.exitQgis()` segfaults in teardown *after* every check has printed; they now
flush stdout and call `os._exit(status)` instead of tearing QGIS down, which is
why the recorded exit codes are 0. A bare `QgsApplication` with no plugin code
init/exit teardown is clean (exit 0), so this is harness teardown, not plugin
code.

### 5.2 e2e QGIS 3.44 LTR output (verbatim, complete)

`C:\OSGeo4W\bin\python-qgis-ltr.bat planx_geostats\tests\qgis_runtime_algorithm_matrix.py --runtime qgis-3.44-ltr`

```
GEOSTATS_RUNTIME_MATRIX: PASS (83/83)
```

Every case `"ok": true`, `failed_count: 0`, runtime `3.44.12-Solothurn`,
`qgis_version_int 34412`. The two new cases reported:

```
=== empirical_semivariogram  ok ===
info: Input diagnostics: 387 valid numeric feature(s), 4 skipped/null/non-numeric feature(s), 391 total feature(s).
info: Binning 74691 point pairs into 12 distance bins up to 23092.4 map units...
info: Fitting the spherical semivariogram model...
info: Fitted spherical: nugget=57.5883, sill=488.75, range=39654.2, R2=0.9682
info: Fitting directional variograms in 8 directions...
info: Writing the fitted-parameter layer...
info: Generating HTML report...

=== kriging_interpolation  ok ===
info: Running leave-one-out cross-validation over 387 observations (one kriging solve each)...
info: Leave-one-out: ME=-0.193439, RMSE=8.31183, MSE=-0.0105472, RMSSE=0.947981
info: Cell size defaulted to 216.375 map units, giving about 200 cells on the longer side.
info: Kriging a 200 x 136 grid (27,200 cells) at 216.375 x 216.603 map units from 387 observations with a none drift.
info: Writing the predicted surface...
info: Writing the kriging variance surface...
info: Generating HTML report...
```

Full log: `qgis_plugins/scratch/logs_geostats/matrix_ltr.txt` (601 lines — the
per-case JSON for all 83 cases plus the `QgsField` deprecation warnings of S12).

### 5.3 e2e QGIS 4.x output (verbatim, complete)

`C:\OSGeo4W\bin\python-qgis.bat planx_geostats\tests\qgis_runtime_algorithm_matrix.py --runtime qgis-4.2`

```
GEOSTATS_RUNTIME_MATRIX: PASS (83/83)
```

Every case `"ok": true`, `failed_count: 0`, runtime `4.2.0-Belém do Pará`.
Full log: `qgis_plugins/scratch/logs_geostats/matrix_42.txt`.

## 6. Gate outputs

The authoritative gate is the Hub scan mirror (bandit + detect-secrets +
flake8 over exactly the files that would ship), because the Hub blocks a version
on any finding and a blocked version number is burned:

```
$ py -3 packaging/hub_security_scan.py planx_geostats

=== planx_geostats  [CLEAN]  212 shipped files ===
  No issues identified -- Hub scan will pass.
============================================================
SUMMARY
  CLEAN     planx_geostats
hub_security_scan exit=0
```

A release zip built from the current tree passes the release-zip verifier
(built into a temp directory; nothing was written to a releases folder, tagged or
uploaded):

```
Wrote planx_geostats-3.9.0.zip (212 files, 2544.1 KB)
verifier exit: 0
VALID RELEASE ZIP
```

`REPORT_v3.9.md` itself never ships: `packaging/zip_hub.py::DEV_DOC_GLOBS`
excludes `REPORT_v*.md` and `ENHANCEMENT_PLAN_*.md` from the zip. Confirmed by
listing the built archive after this report was written — 212 files, unchanged,
with `REPORT in zip: False` and `ENHANCEMENT_PLAN in zip: False`.

Bandit, as the plan defines it (`-r . -x tests,__pycache__`):

```
$ bandit -r . -x tests,__pycache__ -ll
...
	Total issues (by severity):
		Low: 329
		Medium: 13
		High: 0
bandit exit=1
```

**Those 13 are all B608 in `tests/smoke_sample_data.py`, and the plan's
exclusion does not actually exclude them**: bandit's `-x` matches glob patterns
against the path, and `tests` does not match `.\tests\smoke_sample_data.py`. With
a pattern that matches, the shipped code is clean:

```
$ bandit -r . -x './tests/*,*/__pycache__/*' -ll
Code scanned:
	Total lines of code: 33824
Run metrics:
	Total issues (by severity):
		Undefined: 0
		Low: 0
		Medium: 0
		High: 0
bandit exit=0
```

The 13 findings are pre-existing (that file is not in this release's diff) and
are exactly what 3.8.2 ships today; the Hub's own scan of the ship set — the
command that matters — is the `[CLEAN]` result above, which does not include
`tests/`. Corrected exclusion worth folding into the plan for future releases:
`-x './tests/*'`, not `-x tests`.

The `%` gate (ground rule 2), verbatim:

```
$ grep -n "%" planx_geostats/metadata.txt
grep exit=1
```

No literal `%` anywhere in `metadata.txt`.

The three smoke suites and the release-documentation gates:

```
$ py -3 planx_geostats/tests/smoke_core.py            -> CORE SMOKE TESTS OK            (exit 0)
$ py -3 planx_geostats/tests/smoke_sample_data.py     -> SAMPLE DATA SMOKE TESTS OK     (exit 0)
$ py -3 planx_geostats/tests/smoke_provider_catalog.py-> PROVIDER CATALOG SMOKE TESTS OK(exit 0)
```

`smoke_provider_catalog.py` is the gate that carries this release's counters:
`MIN_EXPECTED_ALGORITHM_COUNT = 83`, `EXPECTED_GROUPS` asserted as an exact set
of eight including `07 | Interpolation and Geostatistics`, one registered
algorithm per file, stable ids and groups, dock coverage of every group id, a
unique 64×64 PNG per catalogued algorithm, `HELP_SLUGS` one-to-one with manual
anchors, balanced `<div>` tags, no angle brackets the manual parser would read as
tags, exactly one version string (`v3.9.0`) in the manual, byte-identical root
and `docs/` manual copies, and `## [3.9.0]` / `--version 3.9.0` / `3.9.0 -` in
`CHANGELOG.md` / `README.md` / `metadata.txt`.

Lint over the files this release touches, ignoring E501 (see below):
exit 0. E501 alone is house-wide pre-existing style — 931 occurrences across 80
files, up to 62 in a single file, 24 in the two new ones — all inside embedded
HTML/CSS/SVG string literals, so it is not held against this release.

## 7. Deviations from this plan

1. **`krige_chunked` instead of a plain `krige`.** The plan did not ask for
   chunking; the 18 GB measurement (S8) made it necessary. The public engine
   function is `krige_chunked`, and the harness pins it to `krige` bit-for-bit
   for all three drift orders.
2. **A third core module, `core/raster_io.py`.** The plan did not name it, but
   the raster writing (file writer, block writes, `setEditable(False)`, no-data,
   auto cell size, grid centres) is QGIS-specific and does not belong in the
   numpy-only engine.
3. **`caveats_html` gained an optional argument** (S7) rather than a second
   helper — chosen specifically to keep the 81 pre-existing reports unchanged.
4. **The plan's bandit invocation does not exclude `tests/`** (§6). The
   authoritative Hub scan mirror was used instead, and the corrected pattern is
   recorded.
5. **`REPORT_v3.9.md` is committed alongside the release**, as
   `ENHANCEMENT_PLAN_v3.9.md` already is in this repository; both are stripped
   from the zip by `DEV_DOC_GLOBS`.
6. **The scratch harnesses call `os._exit` instead of `app.exitQgis()`** (§5.1),
   a harness-only change; no plugin code is involved.

Nothing in the plan was skipped. Nothing was released, tagged or pushed.

## 8. Commit list

```
70ffca0         phase-1: add Empirical Semivariogram & Model Fitting algorithm
8fe224d         phase-2: add Ordinary / Universal Kriging Interpolation algorithm
HEAD (the phase-3 commit)
                phase-3: register the interpolation group, sync docs and counters
```

Phase 3 is the commit that carries this report, so its hash cannot be written
into the report itself; read it with `git log -1 --format=%H`. Author and
committer on all three are the maintainer's own identity.

Local commits only. No AI is credited as a contributor on any of them.
