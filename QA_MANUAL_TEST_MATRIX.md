# PlanX GeoStats Lab Manual QA Matrix

Use this matrix before a public upload when changes affect algorithms, reports, sample data, symbology, or model outputs. The Izmir sample is the planning demo fixture; the synthetic QA package is the compact edge-case fixture.

## Setup and Diagnostics

| Tool | Fixture | Parameters | Expected result | Red flags |
|---|---|---|---|---|
| GeoStats Workflow Advisor | none | default HTML output | Report includes planning questions, tool-selection matrix, assumptions, pitfalls, starter recipes, quality gates, and interpretation discipline. | Missing sections, broken HTML, or no Results Viewer output. |
| GeoStats Library Status | active QGIS profile | default HTML output | Lists numpy, scikit-learn, libpysal, esda, spreg, mgwr status and safe install guidance. | Suggests silent install or wrong Python executable. |
| Sample Dataset Guide | bundled samples | load Izmir, synthetic QA, and both | Loads requested layers and writes guide with all QA layer names. | Missing layer, invalid OGR load, or stale layer list. |
| Data Readiness Audit | Izmir sample | core climate/green/network fields | HTML report, CSV/JSON optional outputs, CRS/field/completeness/correlation/workflow findings. | Constant field not flagged, nulls missed, CRS warnings absent. |

## Pattern and Hot Spot Tools

| Tool | Fixture | Parameters | Expected result | Red flags |
|---|---|---|---|---|
| Calculate Distance Band | synthetic `qa_points_grid` | K=5 | HTML report with neighbor-distance summary. | Non-projected unit warning missing or impossible distance values. |
| Global Moran's I | Izmir sample | `median_land_surface_temp_c`, KNN=8 | HTML report with Moran's I, z, p, neighborhood diagnostics. | HTML failure, isolated count mismatch, no CRS/neighborhood diagnostics. |
| Spatial Inequality (Gini and Spatial Gini) | Izmir sample | non-negative field such as `park_m2_per_capita`, KNN=8, permutations=99 | HTML report with classic Gini, neighbor/non-neighbor components, spatial Gini share, polarization, and optional CSV/JSON outputs. | Accepts negative values silently, components do not add to Gini, permutation p-value missing when requested, or neighbor diagnostics absent. |
| Incremental Spatial Autocorrelation | Izmir sample | `median_land_surface_temp_c`, start=500, increment=500, n=10 | Peak distance reported and HTML generated. | `html` shadowing crash or missing peak support summary. |
| General G | Izmir sample | `median_heat_island_index`, KNN=8 | High/low clustering report with caveats. | Fully connected or isolated warning absent when applicable. |
| Getis-Ord Gi* | Izmir sample | `median_land_surface_temp_c`, KNN=8 | Output fields `gi_zscore`, `gi_pvalue`, `gi_conf`, `gi_nbrs`; hot/cold symbology and aliases. | Missing symbology, unaliased fields, null stats for valid records. |
| Local Moran's I | Izmir sample | `median_land_surface_temp_c`, KNN=8 | Cluster/outlier classes and categorized symbology. | HH/LL/HL/LH classes missing or all records not significant unexpectedly. |
| Bivariate Lee's L | Izmir sample | elevation vs. temperature, KNN=5 | Runs on QGIS 3.40+ with `nearestNeighbor` API. | `nearestNeighbors` AttributeError. |

## Geometry and Distribution Tools

| Tool | Fixture | Parameters | Expected result | Red flags |
|---|---|---|---|---|
| Average Nearest Neighbor | synthetic `qa_points_grid` | study area default or explicit | ANN ratio, z, p, HTML report. | Bad area handling or non-finite result. |
| Ripley's K | synthetic `qa_points_grid` | start=250, increment=250, n=3 | K/L-minus-D table and peak interpretation. | Empty distances or geographic CRS warning missing. |
| Mean/Median Center | Izmir sample | optional population weight | Single output feature with center coordinates. | Weight nulls crash or no output feature. |
| Central Feature | Izmir sample | optional population weight | Existing feature marked as central with total distance. | Invalid weight handling or missing source attributes. |
| Standard Distance / Directional Distribution | Izmir sample | optional weight | Circle/ellipse output and stable geometry. | Invalid polygon output or no CRS preservation. |
| Linear Directional Mean | synthetic `qa_lines_directional` | default | Handles single and multipart lines. | Multipart line crash or zero-length trend line. |
| Center/direction output metadata | Izmir and synthetic QA outputs | open result attribute table | Field aliases explain center coordinates, distance/radius, rotation, line count, skipped geometries, and invalid weights. | Output fields are cryptic or missing metadata after post-processing. |

## Modeling Tools

| Tool | Fixture | Parameters | Expected result | Red flags |
|---|---|---|---|---|
| OLS Regression | Izmir sample | temp vs. NDVI, parks, canopy, imperviousness | Residual output, diverging residual symbology, VIF table, model-quality warnings, analyst guidance. | Missing VIF, missing aliases, residual styling absent. |
| GLR Logistic | synthetic `qa_points_grid` | `binary_target` vs. `explanatory_a`, `explanatory_b` | Fitted probabilities in [0,1], diagnostic report. | Accepts non-binary target silently. |
| GLR Poisson | synthetic `qa_points_grid` | `count_target` vs. predictors | Non-negative fitted counts and likelihood with count factorial term. | Negative fitted values or bad AIC. |
| Exploratory Regression | Izmir sample | temp outcome, candidate green/built fields | Ranked candidate models and multicollinearity diagnostics. | Models with too few records not skipped. |
| GWR / MGWR | Izmir sample subset | temp outcome, selected predictors | Local outputs, bandwidth diagnostics, residual map. | Optional dependency message unclear or local coefficient arrays misaligned. |
| Spatial Lag / Spatial Error | Izmir sample | temp outcome, selected predictors, KNN=8 | Spatial parameter, residual diagnostics, styled output. | Islands not warned, PySAL import guidance unclear. |
| Model Comparison Matrix | synthetic model-output layers | `observed_y` | Ranking score, RMSE/MAE/bias, coverage, residual Moran, recommendation. | No rank/score, no residual warning, fails on mixed model outputs. |
| Sensitivity Test | Izmir sample | temp field, multiple simulations | Randomization p-value and neighborhood diagnostics. | Simulation count ignored or report missing caveats. |

## Report Decision Engines

| Area | Fixture | Expected result | Red flags |
|---|---|---|---|
| Workflow Advisor recommendation engine | QGIS-independent smoke test and default Processing run | Goal, geometry, outcome type, and predictor availability produce a personalized sequence, sample suggestions, and combination warnings. | Recommendation logic duplicated back into the algorithm class or missing warnings for mismatched geometry/outcome choices. |
| Model Comparison audit engine | synthetic model-output layers | Score/rank penalizes residual spatial pattern, missing residual diagnostics, and incomplete coverage; recommendation prefers a defensible clean-residual candidate. | Lowest RMSE is always recommended without residual review or ranks are missing. |
| Sensitivity interpretation engine | Izmir sample and sparse synthetic neighborhood scenario | Robust/sensitive verdict, next action, and sensitivity cautions reflect empirical p-value and neighbor graph risk. | Isolated or very dense graphs are treated as fully reliable. |
| Global Moran interpretation engine | Izmir sample and synthetic smoke cases | Clustered/dispersed/random labels, evidence strength, and next action prioritize neighborhood graph quality before local follow-up. | Significant global result is interpreted as site-specific evidence without Local Moran/Gi* follow-up. |
| Spatial Gini decomposition engine | core smoke test and Izmir inequality fields | Classic Gini equals neighbor plus non-neighbor components, spatial Gini share is the non-neighbor share, and polarization reflects average distant-pair versus neighbor-pair difference. | Gini math drifts from pairwise definition or spatial share is reported without neighbor/non-neighbor context. |
| Local pattern class-summary engine | Izmir sample hot spot and LISA outputs | Processing log summarizes hot/cold, cluster/outlier, and dominant class counts; output aliases explain `gi_*` and `lisa_*` fields. | Local tools write styled layers but give no class summary or field metadata. |

## Release Gate

Run these after the manual matrix:

```powershell
py -3 planx_geostats\tests\smoke_core.py
py -3 planx_geostats\tests\smoke_sample_data.py
py -3 planx_geostats\tests\smoke_provider_catalog.py
C:\OSGeo4W\bin\python-qgis-ltr.bat planx_geostats\tests\qgis_runtime_algorithm_matrix.py --root C:\Users\YE\PyCharmMiscProject\qgis_plugins --runtime qgis-ltr
C:\OSGeo4W\bin\python-qgis.bat planx_geostats\tests\qgis_runtime_algorithm_matrix.py --root C:\Users\YE\PyCharmMiscProject\qgis_plugins --runtime qgis4
py -3 packaging\test_verify_release_zip.py
py -3 packaging\validate_plugin.py planx_geostats --strict
```

Only build a release zip after several local improvements have accumulated or when explicitly preparing a Hub upload.
