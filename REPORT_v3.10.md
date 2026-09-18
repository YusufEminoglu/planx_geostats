# Implementation Report v3.10.0

Release: R2 — Interpolation completion (PlanX GeoStats Lab).
Scope: five new Processing algorithms, one new analytical core module, the
reference manual, and the counters/registries that gate them.
Nothing was released, tagged, pushed or uploaded. The maintainer releases
manually after review.

## 1. Release status

Read against `ENHANCEMENT_PLAN_v4.0.md` §3 R2, item by item.

1. **IDW** — **DONE.** `idw_interpolation`, `algorithms/alg_idw.py`. Power,
   three neighbourhood modes (moving window by nearest N, by radius, and every
   observation), convex-hull masking, optional LOO, a neighbour-count raster.
2. **Radial basis function (thin-plate, multiquadric)** — **DONE.**
   `rbf_interpolation`, `algorithms/alg_rbf.py`. Thin-plate and multiquadric
   kernels, an optional smoothing parameter, convex-hull masking, optional LOO.
3. **Natural neighbour (Delaunay)** — **DONE.**
   `natural_neighbour_interpolation`, `algorithms/alg_natural_neighbour.py`.
   Sibson weights over the Delaunay mesh, hull refusal rather than
   extrapolation, a neighbour-count raster.
4. **Global / local polynomial trend** — **DONE.** `polynomial_trend_surface`,
   `algorithms/alg_trend_surface.py`. Orders 1–3, global fit or a
   moving-window refit per neighbourhood (nearest N or radius), a standard-error
   raster, optional LOO.
5. **Indicator kriging (threshold-exceedance probability)** — **DONE.**
   `indicator_kriging`, `algorithms/alg_indicator_kriging.py`, on the new
   `core/indicator_engine.py`. Per-threshold variogram fit, order-relation
   repair, E-type surface and variance, Brier score with a skill score, a
   calibration table, and a declaration of the mapped-versus-sample frequency
   gap with the declustering measurement that explains it.
6. **Empirical Bayesian Kriging** — **DONE.** `ebk_interpolation`,
   `algorithms/alg_ebk.py`. Simulation from a fitted variogram, per-subset
   refit, the chain of fitted models as a layer, prediction and prediction
   variance surfaces, k-fold validation.
7. **Interpolation model comparison** — **DONE.** `interp_compare`,
   `algorithms/alg_interp_compare.py`. One shared seeded protocol, every model
   refitted inside every fold, paired margin against the leader with its
   standard error and its size in standard errors, a CSV and a report, and
   deliberately no raster.
8. **Manual, README, CHANGELOG and metadata in sync** — **DONE.** Two new manual
   cards in both byte-identical copies, sidebar entries, seven cross-linked
   See-also boxes, README catalogue rows and counts, `CHANGELOG.md` and
   `QA_MANUAL_TEST_MATRIX.md`; `metadata.txt` at `3.10.0`.
9. **Counters raised with the tools** — **DONE.** 88 → 90 in the manual (four
   sites), README (five sites) and
   `tests/smoke_provider_catalog.py::MIN_EXPECTED_ALGORITHM_COUNT`, matching 90
   registered algorithms, 90 `HELP_SLUGS` entries, 90 manual cards and 90
   unique 64×64 icons.
10. **Both runtimes green** — **DONE.** 90/90 on QGIS 3.44.12-Solothurn and
    90/90 on QGIS 4.2.0-Belém do Pará (§5).

No item is PARTIAL. No item was skipped.

## 2. Defect outcomes

The defect register this release inherits is §1 of the plan, D1–D12. Items
already closed in R0/R1 are restated with their current state rather than
re-opened; D5 and D9 remain out of scope by the plan's own instruction.

| id | status | note |
|---|---|---|
| D1 | CLOSED (R0) | help-link host pinned by `test_help_url_and_metadata_agree_on_documentation_host`; still green |
| D2 | CLOSED (R0) | per-card `<div>` balance gate; still green with two more cards |
| D3 | CLOSED (R0) | angle-bracket gate; the two new cards pass it |
| D4 | CLOSED (R0), re-satisfied | `test_manual_version_matches_plugin_version` now asserts `v3.10.0` in all four literals |
| D5 | **OPEN — maintainer's** | the published manual is still the retired Pages leftover; redeploying it is a publish |
| D6 | CLOSED (R0), re-satisfied | root and `docs/` manual copies are byte-identical: md5 `dd559a1507867114f4e9f8bfbfcefea4`, 1,195,415 bytes each |
| D7 | CLOSED (R0) | manual source link pinned to `metadata.txt`'s repository |
| D8 | CLOSED (R0) | suite restored to 30/30 in R0 and now carries 36 checks, all green |
| D9 | **OPEN — maintainer's** | `CITATION.cff` untouched; its correct update runs the release pipeline that mints the Zenodo DOI |
| D10 | OPEN, cosmetic | `pf.py` still prints the wrong summary line for QGIS suites; the suite's own JSON is the evidence and was read directly |
| D11 | CLOSED (R0) | `failed_text` no longer reads a failing run as zero failures; both runtime runs reported `failed_count 0` truthfully |
| D12 | documented | the tracker URL 404s; recorded, does not block the Hub |

**New defects found during R2, numbered on from the register:**

**D13 — A predictor's tuple return was reported as a method that "could not be
scored" (medium, latent).** `cross_validate` did `np.asarray(predict(...),
dtype=float)` on the predictor's return value. `core/interpolation_engines.py::idw`
returns `(prediction, used, exact_hits)`, so a caller that handed the engine's
whole return value to the harness produced a ragged three-item array and
`ValueError: setting an array element with a sequence` — a message that names
neither the predictor, nor the fold, nor the real mistake, and the comparison
tool's own error path then reported it as a method that could not be scored,
which reads like a property of the data. Fixed with
`core/interpolation_engines.py::as_prediction`, which validates rank and length
against the fold and raises a message naming the shape, the expected count and
the fold.

**D14 — Multi-band GeoTIFF bands cannot be named on either runtime (low).**
`QgsRasterFileWriter.createMultiBandRaster` exists on 3.44 and on 4.2 and writes
a valid stacked GeoTIFF, but nothing on the returned
`QgsRasterDataProvider` sets a band name, so a multi-band output is bands 1..N to
every reader. `core/raster_io.py::write_float32_multiband_geotiff` therefore
hands the labels back to the caller to publish in the report rather than
pretending they were stored. Recorded as a contract in the module docstring, not
worked around.

**D15 — The indicator report credited the E-type gap to the wrong term (low).**
The report's note attributed the E-type mean's distance from the observed mean
to the "class-midpoint effect". Measured, the whole gap is the mapped-versus-sample
(declustering) difference: with the sample's own marginal class frequencies the
E-type identity returns the observed mean exactly (1.7931·0.2506 + 21.4255·0.2507
+ 30.8315·0.2481 + 39.1842·0.2506 = 23.290 against an observed 23.2891), whereas
the declustered frequencies give 11.912 against a report E-type mean of 11.8342.
The class-representative effect is real but it is the residual between 11.912 and
11.8342, not the gap. The note and the caveat now say so, and the phrase
"class-midpoint effect" is gone from both; the engine returns the class **mean**
(the interval midpoint is only the fallback for an empty class).

**D16 — A latent `NameError` in the indicator report builder (medium, latent).**
A comprehension computing the weak-fit thresholds referenced a name that existed
only as a local of `processAlgorithm`, so report generation could only ever
succeed on a run whose editor never noticed — the line is reached only once a
report is actually written. Hoisted to the module-level `weak_fit_indices(fits)`,
called from both the run log and the report. The flake8 `F821` select is what
surfaced it, which is the argument for keeping that select in the gate.

Two further internal inconsistencies in the same report were found and fixed
with D15/D16 (they are corrections to the same artefact, not separately numbered
defects): the surface statistics were summarised from the unmasked arrays behind
the written rasters while the band table was taken over the written cells, so the
report stated "cells written as no-data: 5,514" beside statistics that included
those cells — both tables now come from the masked grids and agree exactly (mean
exceedance 0.220871 against the band table's 0.2209; E-type mean 11.8342 against
a hand-computed 11.834); and the cross-validation column labelled "mean
predicted exceedance probability" holds the mean of the cumulative
\(P(Z \le z_k)\), not an exceedance probability, and is now labelled as such in
the report, the threshold table and the engine docstring.

## 3. Capability summary

Seven algorithms were added across R2. All are registered under
`07 | Interpolation and Geostatistics` (`planx_interpolation_geostatistics`).

| algorithm id | displayName | inputs | outputs |
|---|---|---|---|
| `idw_interpolation` | Inverse Distance Weighted Interpolation (IDW) | `INPUT`, `FIELD`, `POWER`, `NEIGHBOURHOOD` (all / nearest N / radius), `NEIGHBOURS`, `RADIUS`, `CELL_SIZE`, `MAX_POINTS`, `MASK_OUTSIDE`, `RUN_CV` | `PREDICTION` (Float32 GeoTIFF), `NEIGHBOUR_COUNT`, `HTML_REPORT` |
| `rbf_interpolation` | Radial Basis Function Interpolation (RBF) | `INPUT`, `FIELD`, `KERNEL` (thin plate / multiquadric), `SHAPE`, `SMOOTHING`, `CELL_SIZE`, `MAX_POINTS`, `MASK_OUTSIDE`, `RUN_CV` | `PREDICTION`, `HTML_REPORT` |
| `natural_neighbour_interpolation` | Natural Neighbour Interpolation (Sibson) | `INPUT`, `FIELD`, `CELL_SIZE`, `MAX_POINTS`, `MASK_OUTSIDE`, `RUN_CV` | `PREDICTION`, `NEIGHBOUR_COUNT`, `HTML_REPORT` |
| `polynomial_trend_surface` | Polynomial Trend Surface Interpolation | `INPUT`, `FIELD`, `ORDER` (1–3), `NEIGHBOURHOOD` (global / nearest N / radius), `NEIGHBOURS`, `RADIUS`, `CELL_SIZE`, `MAX_POINTS`, `MASK_OUTSIDE`, `RUN_CV` | `PREDICTION`, `STD_ERROR`, `HTML_REPORT` |
| `ebk_interpolation` | Empirical Bayesian Kriging Interpolation | `INPUT`, `FIELD`, `MODEL`, `ALPHA`, `DRIFT`, `N_SIMS`, `SUBSET_SIZE`, `OVERLAP`, `SEED`, `N_BINS`, `CELL_SIZE`, `MAX_POINTS`, `RUN_CV`, `CV_FOLDS`, `MASK_OUTSIDE` | `PREDICTION`, `VARIANCE`, `SUBSET_LAYER` (per-subset fitted model), `HTML_REPORT` |
| `indicator_kriging` | Indicator Kriging (Threshold Exceedance Probability) | `INPUT`, `FIELD`, `THRESHOLDS`, `THRESHOLD_COUNT`, `MODEL`, `ALPHA`, `N_BINS`, `MAX_LAG`, `CELL_SIZE`, `MAX_POINTS`, `RUN_CV`, `CV_FOLDS`, `MASK_OUTSIDE` | `EXCEEDANCE` (one band per threshold), `ETYPE`, `ETYPE_VARIANCE`, `HTML_REPORT` |
| `interp_compare` | Compare Interpolation Methods | `INPUT`, `FIELD`, `METHODS`, `IDW_POWER`, `RBF_SMOOTHING`, `KRIGING_MODEL`, `ALPHA`, `N_BINS`, `MAX_LAG`, `SEED`, `MAX_POINTS` | `CSV_TABLE`, `HTML_REPORT` — no raster, by design |

New core module: `core/indicator_engine.py` (numpy only). New engine helpers in
`core/interpolation_engines.py`: `idw`, `rbf`, `sibson`, `polynomial_trend`,
`cross_validate`, `as_prediction`, `kfold_indices`. New raster helper in
`core/raster_io.py`: `write_float32_multiband_geotiff`.

## 4. Validation evidence

**IDW, RBF and natural neighbour produce real surfaces on the bundled İzmir FUR
sample** (387 usable observations of `road_density`, a city-wide density field
where a smooth interpolator is the wrong model and the numbers say so, which is
the honest result rather than a flattering one):

```
IDW     : 200 x 136 grid (27,200 cells) at 216.375 x 216.603 map units, power 2;
          5,514 of 27,200 cells (20.3%) fell outside the convex hull and were
          masked to no-data.
RBF     : leave-one-out ME=-0.497109, RMSE=7.77059, MAE=5.79453.
NN      : 12 of 387 observations could not be scored — they lie on the convex
          hull, so once held out they fall outside the hull of the rest.
          87 x 59 grid (5,133 cells); 4,095 inside the hull were interpolated,
          1,038 (20.2%) outside carry no prediction.
```

**Empirical Bayesian Kriging validates against its own reported variance.**

```
5-fold, 387 observations, 30 simulations, subset size 100, each fold refitting
every subset:
    ME=-0.689815, RMSE=9.13104, MSE=-0.0581002, RMSSE=1.00006
    fitted family per subset: exponential, gaussian
    effective model count 1.64 of 30 simulations
```

RMSSE = 1.00006 against an ideal of 1.0 is the number that matters here: the
prediction variance EBK reports is the spread the method actually produces,
*including* the variogram-estimation uncertainty that the plug-in kriging
variance leaves out. A method that quoted only the kriging variance would report
a smaller number and a worse RMSSE.

**Indicator kriging, on the İzmir fixture, default three thresholds.**
The fit per threshold, quoted from the report:

```
threshold 1 (12.9765): gaussian, range 22656.3, R² 0.9880
threshold 2 (27.1038): spherical, range 11261.4, R² 0.9549
threshold 3 (34.3258): spherical, range   981.406, R² 0.0000
```

The order-relation repair and the probability validation are both real numbers,
not assertions:

```
order-relation repair: 11,679 of 81,600 probability cells clipped into [0, 1];
                       20,124 of 81,600 repaired; up to 3 adjacent thresholds
                       pooled into one; largest single move 0.207545
Brier score         : 0.125997 against a base-rate score of 0.208548
Brier skill score   : +0.3958 over 1,161 held-out scores
calibration (mean predicted vs observed, 5 equal-count bins per threshold):
                        0.2445 vs 0.2506, 0.4965 vs 0.5013, 0.7295 vs 0.7494
                        widest calibration gap 0.0543
```

**The number in the indicator report that looks like a bug, and the measurement
that shows it is not.** The bands' mean probability over the mapped area is
0.6170 / 0.8228 / 0.8976 against sample frequencies 0.2506 / 0.5013 / 0.7494.
That is ordinary kriging's far field converging on the *declustered* mean, and it
was measured rather than argued. At the top threshold the fitted range is
**981.4061** map units while far targets sit 17,924 map units from the nearest
observation, so γ(d) is the sill for every observation:

```
sill 0.176763 at every observation; the kriging system's right-hand side is the
same constant for all 387 of them
weights over 36 off-data targets on rings 1x / 2x / 4x the box span:
    min 0.0000, max 0.0038, against an equal weight of 0.002584
condition number 2.006e3 — well determined, not ill-conditioned
estimate 0.858195 at every one of the 36 targets
observed indicator frequency 0.749354 != 0.858195
```

The same estimate at all 36 targets on three different rings is the point: the
values enter only through the weights, and the weights are a property of where
the observations are. Cell declustering over a sweep of cell sizes
(`DECLUSTER_CELL_FRACTIONS`, 0.05–0.75 of the box span, giving cell sizes
2163.75 to 32456.2 map units) gives cumulative frequencies of 0.5467 to 0.9504
across the sweep with medians 0.6048, 0.8128 and 0.9283, and the tool's mapped
band means (0.6170 / 0.8228 / 0.8976) sit on those medians rather than on the
sample frequencies — which is the explanation, not a coincidence. The busiest
cell holds 359 observations. That is why the report prints the sweep's range and
median rather than one nominated cell size.

The E-type algebra, verified numerically:

```
sample marginal frequencies : 1.7931*0.2506 + 21.4255*0.2507 + 30.8315*0.2481
                            + 39.1842*0.2506 = 23.290   vs observed mean 23.2891
declustered frequencies     : 11.912 vs report E-type mean 11.8342
```

So the E-type gap is the mapped-versus-sample distinction almost entirely, and
the class-representative effect is only the residual (11.912 → 11.8342).

**Interpolation model comparison.** Two runs, both real, different protocols —
stated because a reviewer re-running either will get one of them:

*Matrix run*, algorithm defaults (4 methods: ordinary kriging, IDW, RBF,
natural neighbour; k-fold):

```
RBF (thin plate) RMSE 8.04619, MAE 6.07073, ME -0.481378 over 387 scored
Ordinary kriging RMSE 9.66462, MAE 7.55215, ME -0.0864443 over 387 scored
not every method could predict every held-out observation, so the ranking is
computed on the 375 observations all 4 scored
Natural neighbour against RBF: mean difference in absolute error -0.394229
    with a standard error of 0.176781, 2.2 standard errors
```

*Seeded 10-fold run* (`scratch/run_r2_tools.py`, one seed, 10 folds, all six
methods), which is the run the manual card and the CHANGELOG quote:

```
rank method              rmse        margin_vs_leader  se        in SE
1    Natural neighbour   7.428914253  (leader)          -         -
2    RBF (thin plate)    7.742377079  -0.3211155149     0.1777521 1.8065
3    Ordinary kriging    9.801871502  -2.132309773      0.2607935 8.1762
4    IDW                 11.04161047  -2.769480567      0.3649772 7.5881
374 of 387 observations compared; 13 dropped by natural neighbour
```

Both runs agree on the finding: the top two are not separable (1.8 SE, swapping
places on 50.8 / 49.2 per cent of locations), and the third and fourth trail by
8.2 and 7.6 SE, which are findings. The card's "1.8 standard errors" and the
matrix's "2.2 standard errors" are the same conclusion from two protocols, and
the report's own verdict — *this sample does not separate them* — is what both
produce.

**Byte-identical outputs across runtimes.** All six R2 tool products from the
matrix harness on both runtimes:

```
07_compare.csv                    9a5731ca0b8f05f2987a0d0b3f220af7
07_compare.html                   af3d49eb385cff16e4bdced0d3f0f70b
07_indicator.html                 ac96044b615e9c3f9fbf4c886a0352a4
07_indicator_etype.tif            2a9fb055163abc418d221ac1e6f3b1c0
07_indicator_etype_var.tif        8214fc220797f1150da5c251bc4e867d
07_indicator_exceedance.tif       82ed25f0688081200dc7ab1b8531e317
```

The indicator report was regenerated after the D15 wording fix and re-checked in
the output rather than only in the source: `class-midpoint` occurs **0** times,
`class-representative` **1** time.

## 5. Test evidence

### 5.1 Pure suites (verbatim)

```
$ py -3 planx_geostats/tests/smoke_core.py
CORE SMOKE TESTS OK
exit 0

$ py -3 planx_geostats/tests/smoke_sample_data.py
SAMPLE DATA SMOKE TESTS OK
exit 0

$ py -3 planx_geostats/tests/smoke_provider_catalog.py
PROVIDER CATALOG SMOKE TESTS OK
exit 0
```

### 5.2 QGIS 3.44.12-Solothurn (LTR), verbatim

```
$ QGIS_CUSTOM_CONFIG_PATH=$(mktemp -d) C:/OSGeo4W/bin/python-qgis-ltr.bat \
      planx_geostats/tests/qgis_runtime_algorithm_matrix.py
GEOSTATS_RUNTIME_MATRIX_JSON={"runtime": "unknown", "qgis_version":
  "3.44.12-Solothurn", "qgis_version_int": 34412, ...
  "case_count": 90, "ok_count": 90, "failed_count": 0, ...}
GEOSTATS_RUNTIME_MATRIX: PASS (90/90)
```

### 5.3 QGIS 4.2.0-Belém do Pará, verbatim

```
$ QGIS_CUSTOM_CONFIG_PATH=$(mktemp -d) C:/OSGeo4W/bin/python-qgis.bat \
      planx_geostats/tests/qgis_runtime_algorithm_matrix.py
GEOSTATS_RUNTIME_MATRIX_JSON={"runtime": "unknown", "qgis_version":
  "4.2.0-Bel\u00e9m do Par\u00e1", "qgis_version_int": 40200, ...
  "case_count": 90, "ok_count": 90, "failed_count": 0, ...}
GEOSTATS_RUNTIME_MATRIX: PASS (90/90)
```

### 5.4 The new in-repo coverage, and its mutation check

`tests/smoke_core.py` gained `load_core_package`, which loads the geostatistics
engines into a synthetic package so their relative imports resolve without a QGIS
runtime, and `test_declustered_frequencies_discount_a_crowded_cluster`, which
asserts that a scattered sample decomposes to its own frequencies (within 0.05),
that a 90-into-1-square crowd is discounted rather than restated
(< 0.5 × 0.90), that the answer follows the values and not the geometry (flipping
which cluster holds the low value moves the median across 0.5 in the other
direction), that a sample with no spatial extent falls back to the plain
frequency, and that an empty threshold list is refused.

The test was **mutation-checked**, not merely run: stubbing
`declustered_frequencies` with one that restates the sample frequency raises

```
AssertionError: the crowd was restated rather than discounted:
plain 0.9000 against a declustered median of 0.9000
```

so the assertion bites. A test that passes against both a correct and a broken
implementation proves nothing.

## 6. Gate outputs

The authoritative gate is the Hub scan mirror, because the Hub blocks a version
on any finding and a blocked version number is burned:

```
$ py -3 packaging/hub_security_scan.py planx_geostats

=== planx_geostats  [CLEAN]  231 shipped files ===
  No issues identified -- Hub scan will pass.
============================================================
SUMMARY
  CLEAN     planx_geostats
hub_security_scan exit=0
```

```
$ py -3 packaging/pf.py verify planx_geostats

=== planx_geostats - verify  [OK] ===
  [PASS] validate_plugin - 38 warning(s)
  [PASS] Hub gate: security scan (blocking) - CLEAN
  [PASS] Qt6 compatibility (advisory) - no errors; forward-compat advisories only
  [PASS] pure: tests/smoke_core.py
  [PASS] pure: tests/smoke_sample_data.py
  [PASS] pure: tests/smoke_provider_catalog.py
  [PASS] qgis[ltr]: tests/qgis_runtime_algorithm_matrix.py - [WinError 2] ...
  [PASS] qgis[qgis4]: tests/qgis_runtime_algorithm_matrix.py - [WinError 2] ...

============================================================
VERIFY OK: 1 plugin(s)
```

The two `[WinError 2]` lines are the known **D10** artefact — `pf.py` reports the
wrong summary line for every QGIS suite, on every plugin in this repository. The
suite's own JSON in §5 is the authoritative result; §5.2 and §5.3 were run
directly rather than through `pf.py` for exactly that reason.

The `%` gate (ground rule 2), verbatim:

```
$ grep -n "%" metadata.txt
grep exit=1
```

No literal `%` anywhere in `metadata.txt`.

The flake8 CI select, verbatim:

```
$ python -m flake8 . --select=E901,E902,E999,F821,F823,F831,E741,E742,E743 \
      --max-line-length=1000 \
      --exclude=.git,__pycache__,tests,e2e,scratch,node_modules
flake8 exit=0
```

No output. This is the gate that caught D16: the latent `NameError` in the
indicator report builder is an `F821` and nothing else would have seen it, since
the line is only reached when a report is actually written.

`validate_plugin`'s 38 warnings are the pre-existing CRLF item (**S11** in
`REPORT_v3.9.md`), unchanged in count by this release and out of scope.

## 7. Deviations from this plan

1. **The manual card's comparison figures and the runtime matrix's comparison
   figures come from two different protocols.** The card quotes the seeded
   10-fold run in `scratch/run_r2_tools.py` (1.8 SE for the top pair, 374 of 387
   compared); the matrix harness runs the algorithm with its own defaults (2.2 SE,
   375 of 387). Both are reproduced in §4 rather than one being presented as the
   other. The card names the protocol generically ("one shared split, one seed")
   and does not state the seed value; a reviewer re-running the *matrix* will
   reproduce the second set, not the card's.

2. **Some numbers quoted in the shipped manual were measured by scratch
   harnesses that live outside this repository**, and are therefore not an
   in-repo gate a reviewer can re-run from a clone:
   `scratch/probe_farfield.py` (the far-field weights, the condition number,
   the constant estimate at 36 targets) and `scratch/verify_indicator.py` (nine
   indicator checks). The declustering behaviour they support is now also covered
   in-repo by §5.4; the far-field measurement specifically is not. Stated here
   because a number a reviewer cannot reproduce is a number they are entitled to
   discount.

3. **`core/raster_io.py` gained a third public writer,
   `write_float32_multiband_geotiff`**, which the plan did not name. It is what
   `indicator_kriging` needs to write one band per threshold, and D14 is the
   constraint it carries.

4. **`core/interpolation_engines.py::cross_validate` changed signature
   behaviour** (D13): it now rejects a predictor that does not return one float
   per test location instead of propagating a numpy shape error. No caller
   relied on the old behaviour — passing the engine's tuple was always a bug.

5. **`interp_compare` writes no raster at all**, which is a deliberate reading of
   "mirroring ArcGIS's *Compare Geostatistical Layers*": the tool's job is the
   verdict, and it names the tool that produces the winner's surface instead of
   duplicating it.

6. **`REPORT_v3.10.md` is committed alongside the release**, as
   `REPORT_v3.9.md` already is in this repository; both are stripped from the
   release zip by `packaging/zip_hub.py::DEV_DOC_GLOBS`.

7. **D5 and D9 are untouched**, per the plan. Nothing in this release
   redeploys the live manual or regenerates `CITATION.cff`.

Nothing in the plan was skipped. Nothing was released, tagged or pushed.

## 8. Commit list

```
fb94c11         R2 phase-1: interpolation engines, IDW, RBF and polynomial trend surface
7d713fb         R2 phase-2: Sibson natural neighbour interpolation, and a no-data fix
7dd22c6         R2 phase-3: Empirical Bayesian Kriging, and a validation limit that refused it
HEAD (this commit)
                R2 phase-4: indicator kriging, interpolation comparison, and the counters at 90
```

Phase 4 is the commit that carries this report, so its hash cannot be written
into the report itself; read it with `git log -1 --format=%H`. Author and
committer on all four are the maintainer's own identity.

Local commits only. No AI is credited as a contributor on any of them.
