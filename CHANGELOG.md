# Changelog

## [2.6.0] - 2026-08-19

- Five more reports gain charts: Random Forest, Extra Trees, and Support Vector classification reports each gain a shaded heatmap of their confusion matrix (sequential single-hue, colorblind-safe) alongside the existing table; Partial Dependence Report gains a line chart of its sweep curve; Conformal Prediction Interval gains a sorted-prediction interval plot (shaded lower/upper ribbon with point predictions), deterministically subsampled to ~1,500 records on very large layers. core/charts.py::line_chart_svg() gained an optional band parameter for the ribbon plot, covered by its own smoke-test case. Manual entries updated.

## [2.5.0] - 2026-08-19

- Five more reports gain charts: Ripley's K-Function gets a line chart of observed vs. CSR-expected K(d) across distance bands with the peak L(d)-d departure marked; Spatial Gini gains a Lorenz curve with the classic Gini labeled; ML Model Comparison and Model Comparison Matrix both gain a ranked leaderboard bar chart (best model highlighted); Spatial k-Fold Cross-Validation Evaluator gains a per-fold metric bar chart. All five charts are computed directly from data these tools already extract or compute - no new statistics. Manual entries updated.

## [2.4.0] - 2026-08-19

- Four more spatial-autocorrelation reports gain charts: Geary's C gets a permutation-histogram (observed C marked), reusing the same permuted values the statistic already computes; Model Residual Spatial Autocorrelation Check gains a residual Moran scatterplot; Global Bivariate Lee's L gains a bivariate Moran-style scatterplot (standardized Field X vs. spatial lag of standardized Field Y); Incremental Spatial Autocorrelation's existing hand-rolled correlogram was refactored onto the shared line_chart_svg() helper (same visual output). core/analysis_diagnostics.py and core/advanced_stats_engines.py now expose the per-point/permutation arrays these charts need alongside their existing summary statistics. Manual entries updated.

## [2.3.0] - 2026-08-19

- First three HTML reports now draw the inline-SVG charts core/charts.py added last release: Global Moran's I gets a proper Moran scatterplot (standardized value vs. spatial lag, OLS trend line, HH/HL/LH/LL quadrant shading); Attribute Randomization Sensitivity Test's existing hand-rolled histogram was refactored to use the shared histogram_svg() helper (same visual output, now shared code); Permutation Feature Importance gained a ranked bar chart above its table. Manual entries updated to describe the new charts.

## [2.2.0] - 2026-08-19

- Add core/charts.py, a new dependency-free inline-SVG chart toolkit (bar/scatter/line/histogram/heatmap/rose/Lorenz charts plus KPI dashboard cards) that HTML reports will draw on starting next release. Stdlib-only (html, math), no matplotlib/PIL/network calls, enforced by a new AST-based smoke test alongside structural SVG-validity and edge-case coverage in tests/smoke_provider_catalog.py. No algorithm report output changed yet - this release ships the shared toolkit only, per VISUALIZATION_PATHWAY.md (now committed) increment 1.

## [2.1.2] - 2026-08-15

- **Bug fix**: Conformal Prediction Interval called `mapie.regression.MapieRegressor`, an API removed in `mapie` 1.0's rewrite (the package installs as `mapie` 1.x by default today). Every run failed with an import error once `mapie` was actually installed. Rewrote the wrapper (`core/ml_engines.py::fit_conformal_interval`) against the current `CrossConformalRegressor`/`fit_conformalize`/`predict_interval` API, preserving the same jackknife+ cross-conformal method and coverage semantics; pinned `mapie>=1.0` in `requirements_geostats.txt`.
- **Bug fix**: TabPFN Regression/Classification crashed with a raw `OSError: [WinError 10038]` instead of a usable error whenever the one-time TabPFN license/model-download step had not already been completed, because TabPFN's interactive browser-login prompt polls `sys.stdin` with `select.select()`, which only supports sockets on Windows. This is an upstream TabPFN limitation on Windows, not fixable from the plugin side, so both TabPFN tools now catch the failure and raise a clear, actionable message: accept the license once at https://ux.priorlabs.ai/account outside QGIS, then set a permanent `TABPFN_TOKEN` environment variable so TabPFN authenticates silently on every future run with no browser prompt.
- Added real QGIS runtime execution coverage for **06 | Machine Learning and Explainable AI** (all 34 algorithms) to `tests/qgis_runtime_algorithm_matrix.py`, which previously exercised Groups 00-05 only. 19 tools that need only scikit-learn now run for real against the sample data on every verify; the 15 that need an optional package or TabPFN's license step correctly report that condition instead of failing. This is how both bugs above were actually found, rather than assumed fixed.

## [2.1.1] - 2026-08-15

- **Bug fix**: the docked GeoStats Lab panel's group list (`geostats_dock.py::GEOSTATS_GROUPS`) never included `06 | Machine Learning and Explainable AI`. All 34 algorithms in that group — including every tool added in 2.1.0 (Conformal Prediction Interval, TabPFN, DiCE, CatBoost, EBM) plus the original 26 from 2.0.0 — were fully registered and runnable from the Processing Toolbox the entire time, but silently never appeared in the docked panel specifically, since the panel iterates its own hardcoded group list rather than reading group ids directly off the provider. Fixed by adding the missing group entry, and added `tests/smoke_provider_catalog.py::test_dock_group_list_covers_every_algorithm_group_id`, which now fails the build if any future group is ever left out of the dock's list again.

## [2.1.0] - 2026-08-15

- **8 new algorithms, taking the plugin from 73 to 81**, all in **06 | Machine Learning and Explainable AI**:
  - **Conformal Prediction Interval** — distribution-free prediction intervals with a proven marginal coverage guarantee (MAPIE's jackknife+ cross-conformal method), for any regression model in the group, not just Random Forest/Extra Trees.
  - **TabPFN Regression** and **TabPFN Classification** — a 2025 zero-shot tabular foundation model (Hollmann et al., *Nature*, 2025): a transformer pretrained once, offline, on millions of synthetic datasets, performing in-context learning at inference time with no per-dataset training loop or hyperparameters.
  - **DiCE Counterfactual Explanation** — the action-oriented complement to SHAP: diverse, minimal field-value changes to one record that would flip its predicted class (Mothilal, Sharma & Tan, ACM FAT* 2020).
  - **Gradient Boosting Regression/Classification (CatBoost)** — a 4th GBM engine alongside scikit-learn/XGBoost/LightGBM, using ordered boosting to remove the target-leakage bias present in classical gradient boosting (Prokhorenkova et al., NeurIPS 2018).
  - **Explainable Boosting Machine Regression/Classification** — a glass-box generalized additive model (Lou, Caruana & Gehrke, KDD 2012) whose per-field contribution to every prediction is exact, read directly off the fitted model, not approximated by sampling the way SHAP explains a black-box model.
- **Spatial k-Fold Cross-Validation Evaluator** gained a **kNNDM** fold-assignment option (Linnenbrink, Milà, Ludwig & Meyer, *Geoscientific Model Development*, 2024) alongside the existing K-Means spatial block method — chooses the fold split whose induced test-to-train distance distribution best matches the dataset's own leave-one-out distance distribution, rather than optimizing purely for geographic compactness.
- New optional dependencies (`catboost`, `interpret`, `mapie`, `dice-ml`, `tabpfn`) install through the existing Setup and Diagnostics > Install / Update GeoStats Libraries workflow.
- Reference manual expanded with full Theoretical Background / Mathematical Formulation / Parameters / Output / Interpretation Guide / Literature entries for all 8 new tools, and deepened across the 26 pre-existing Machine Learning entries with additional theory paragraphs, equations, interpretation guidance, and citations.
- Fixed two pre-existing bugs in Multiscale Geographically Weighted Regression (MGWR), unrelated to this release's new tools, surfaced while re-running the full QGIS runtime verification gate: an adaptive-kernel bandwidth-search precondition that previously failed with an opaque numpy error on small samples now raises a clear, actionable message instead; and a result-extraction crash when `hat_matrix=False` (a property access that raised instead of returning a safe default).

## [2.0.0] - 2026-08-15

- **29 new algorithms, taking the plugin from 44 to 73**, across one new group plus one existing group:
  - **06 | Machine Learning and Explainable AI (26 tools)**: Random Forest, Extra Trees, Support Vector, and Neural Network (MLP) regression/classification; Gradient Boosting regression/classification across three engines (scikit-learn `HistGradientBoosting`, XGBoost, LightGBM — six tools total); Spatial k-Fold Cross-Validation Evaluator (K-Means-blocked folds, not random shuffling, to remove spatial-autocorrelation leakage); Permutation Feature Importance; Partial Dependence Report; ML Model Comparison (Leaderboard); **SHAP Global Feature Importance**, **SHAP Spatial Attribution Map** (writes every explanatory field's per-record SHAP contribution back onto the map as a symbolizable `shap_<field>` column — this release's flagship capability), and **SHAP Local Explanation Report**; Model Residual Spatial Autocorrelation Check (Global Moran's I on a fitted ML model's residuals); Prediction Uncertainty Map (per-tree spread for Random Forest/Extra Trees); DBSCAN, HDBSCAN, and Gaussian Mixture Model clustering.
  - **05 | Models and Scenarios (+3 tools)**: Spatial Regime Regression (`spreg.OLS_Regimes` with a joint Chow test for structural instability across regimes), Quantile Regression (hand-rolled iteratively-reweighted-least-squares on the pinball loss — no new dependency), Geographically Weighted Summary Statistics (local mean/std/skew via the same GWR/MGWR kernel families, plus Kish effective sample size).
- New optional dependencies (`xgboost`, `lightgbm`, `shap`) install through the existing Setup and Diagnostics > Install / Update GeoStats Libraries workflow, alongside `numba`/`libpysal`/`esda`/`spreg`/`mgwr`/`scikit-learn`.
- A new synthetic classification QA/demo GeoPackage (`planx_geostats_classification_qa`) fills the gap the bundled Izmir FUR sample cannot cover on its own — it has no categorical field. Sample Dataset Guide and Workflow Advisor updated to load and recommend it.
- Reference manual expanded with a full Theoretical Background / Mathematical Formulation / Parameters / Output / Interpretation Guide / Literature entry for every new tool (44 to 73 manual entries), each with real academic citations. Manual hero stats, sidebar navigation, and workflow diagram updated for 7 analytical groups.
- Network centrality and accessibility tools (Network Betweenness/Closeness/Straightness Centrality, Network Reach, 2SFCA, Gravity-Based Accessibility, Nearest-Facility Coverage Gap) were built, then removed before release as out of scope for a spatial-statistics lab - that functionality already lives in the main PlanX plugin.

## [1.0.0] - 2026-08-13

- New **İzmir Functional Urban Region (FUR)** street-network / space-syntax sample dataset (391 features, 34 real fields — `road_density`, `betweenness_mean`, `ss_integration_median`, `gridiron_index`, `transit_accessibility`, and more), replacing the old population/heat-map sample. Every example workflow, the Workflow Advisor's recipes, the Sample Dataset Guide, and the manual's own walkthroughs were rewritten to match the new data, grounded in real computed statistics (not assumed).
- Added **10 new advanced spatial-statistics tools**, taking the plugin from 34 to 44 algorithms:
  - **02 | Urban Pattern Scan**: Geary's C, Join Count Statistics (BB/WW/BW), Global Bivariate Lee's L, Geodetector Q-Statistic (spatial stratified heterogeneity).
  - **03 | Hot Spots and Spatial Outliers**: Local Geary's C, Colocation Quotient (CLQ), SKATER spatially constrained regionalization (shipped as a native NumPy + Prim's-algorithm implementation, no optional dependency required).
  - **05 | Models and Scenarios**: Lagrange Multiplier Diagnostics (the formal SAR-vs-SEM specification test), Spatial Durbin Model (Spatial 2SLS), Eigenvector Spatial Filtering Regression.
  - All 4 new global statistics and the local/colocation tools use Monte Carlo permutation inference rather than closed-form variance formulas that are easy to get subtly wrong; the Lagrange Multiplier Diagnostics engine was independently validated against a hard mathematical identity (LM-lag + Robust LM-error = LM-error + Robust LM-lag) plus directional checks against known synthetic spatial-lag and spatial-error processes.
  - Each new tool ships with a full elite-depth reference-manual entry (Theoretical Background, Mathematical Formulation, Parameters, Output, Interpretation Guide, Literature) and a unique algorithm icon.
- Redesigned the main plugin icon: the previous teal-gradient rings/dots/grid motif collapsed into an unreadable smear at real QGIS toolbar sizes (16-24px). Replaced with a bolder spatial-weights network glyph, verified legible at 16px.
- Manual hero stats, README tool catalog, and release-gate test thresholds updated for 44 algorithms and 300+ citations.

## [0.10.0] - 2026-08-13

- Deepened every algorithm's in-GUI help text (`shortHelpString`) with concrete field-by-field interpretation guidance, thresholds, and cross-tool recommendations — previously 30-150 words each, now 180-310 words.
- Added a genuine Theoretical Background section to the six Centers, Direction and Dispersion algorithms' manual entries, grounded only in each entry's own already-cited literature.
- Fixed broken `helpUrl()` deep links on all 34 algorithms (manual anchor slugs never matched `name()`-derived hashes), a wrong icon on Bivariate LISA, a manual HTML tag-balance bug, 4 mismatched algorithm-id labels, and 1 broken DOI link.
- Added a dockable **GeoStats Lab** panel (toolbar icon + Plugins menu entry) grouping all 34 tools by category with search and one-click launch. The plugin is now packaged as a hybrid Processing + dock-GUI plugin (it was Processing-only since 0.9.1).
- Added a copy-anchor-link button, 24 "See also" cross-reference boxes, and 5 performance/complexity badges to the reference manual.

## [0.9.23] - 2026-08-08

- Update documentation link to GEOSTATS_REFERENCE_MANUAL.html

## [0.9.22] - 2026-08-07

- Added online user manual link (https://yusufeminoglu.github.io/planx_geostats/GEOSTATS_REFERENCE_MANUAL.html) and GitHub repository star call-to-action.

## [0.9.21] - 2026-08-07

- Add floating Save as PDF button to reference manual

## [0.9.20] - 2026-08-07

- 2.5x manual expansion: 300+ equations, 215+ DOI refs, HelpUrlMixin, GitHub Pages

## [0.9.19] - 2026-07-14

- Add Bivariate Local Moran's I (Bivariate LISA) algorithm and unique icon

## [0.9.18] - 2026-06-18

- docs: add CITATION.cff for Zenodo DOI integration

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.9.17] - 2026-06-05

- Added `Spatial Inequality (Gini and Spatial Gini)` to the Urban Pattern Scan group.
- Added classic Gini, neighbor/non-neighbor Gini components, spatial Gini share, spatial polarization, and permutation inference.
- Added optional CSV/JSON exports, HTML analyst guidance, algorithm icon, provider registration, smoke coverage, and release-gate count alignment.
- Added a full QGIS runtime algorithm matrix that executes every GeoStats Processing algorithm against bundled sample data on QGIS 3 LTR and QGIS 4.
- Restored the root plugin icon path by packaging `icons/icon.png` and pointing `metadata.txt` to the PNG asset used by the Processing provider.
- Fixed Similarity Search expression-context handling and geometry-type checks in OLS/GLR residual diagnostics.
- Hardened optional PySAL/MGWR dependency diagnostics so broken import-time dependency stacks are reported with the exact installer command.

## [0.9.15] - 2026-05-30

- Revised plugin icon to SVG

## [0.9.14] - 2026-05-27

- Code quality and Hub submission hygiene.
  - Added setup.cfg (flake8: W503, E203 disabled; max-line-length=120).
  - Fixed 21 flake8 issues across 15 files: unused QVariant imports, missing blank lines, continuation indentation hoisted from f-strings, trailing whitespace stripped.
  - All 42 Python files compile cleanly. Zero non-E501 flake8 issues.

## [0.9.13] - 2026-05-26

- GeoStats provider/runtime smoke coverage updates and QGIS 3.40+/4 compatibility validation.

## [0.9.12] - 2026-05-26

- Maintenance release: refreshed Plugin Hub package after QGIS 3 and QGIS 4 compatibility validation.

## [0.9.10] - 2026-05-22

### Changed
- Completed beta stabilization for the PlanX GeoStats Lab Processing provider.
- Centralized optional dependency error guidance for MGWR, Spatial Lag Regression, and Spatial Error Regression so missing-library failures point users to GeoStats Library Status and the explicit installer workflow.
- Hardened output-layer metadata handling so field descriptions are retained as custom layer properties even when a provider cannot apply field aliases.
- Hardened centroid and geometry handling across distance, center, distribution, pattern, GWR, MGWR, and export workflows so missing, empty, invalid, or provider-problem geometries are skipped safely instead of crashing the algorithm.

### Added
- Added QGIS-independent smoke coverage for dependency guidance, metadata resilience, null/empty geometry handling, and centroid failure handling.
- Added release-gate coverage to keep core decision helpers, output metadata contracts, manual QA documentation, and package verification aligned.

## [0.9.9] - 2026-05-22

### Added
- Added personalized Workflow Advisor inputs for analysis goal, geometry context, outcome type, and explanatory-variable availability.
- Added a personalized recommendation section that returns a suggested tool sequence and pre-trust checks.

## [0.9.8] - 2026-05-22

### Added
- Added GeoStats Workflow Advisor, a Processing report that maps planning questions to recommended GeoStats tools, prerequisites, outputs, and follow-up checks.
- Added method assumptions, common pitfalls, safer moves, and bundled-sample starter recipes to the Workflow Advisor report.
- Added smoke coverage for Workflow Advisor guidance sections and a minimum provider algorithm-count guard.
- Added a manual QA matrix covering setup, pattern analysis, geometry summaries, modeling workflows, symbology, reports, and release gates.
- Added shared report-guidance and output-layer metadata helpers for more consistent professional reports and layer outputs.
- Added VIF diagnostics to regression quality reporting and rank/score auditing to Model Comparison Matrix.

## [0.9.7] - 2026-05-22

### Added
- Added smoke coverage that keeps `metadata.txt`, `CHANGELOG.md`, and the README release verification command synchronized on the same plugin version.

## [0.9.6] - 2026-05-22

### Changed
- Expanded the Sample Dataset Guide QA fixture table to list every model-output layer explicitly instead of using a wildcard label.

### Added
- Added smoke coverage that verifies the generated Sample Dataset Guide source mentions every loadable sample and QA layer.

## [0.9.5] - 2026-05-22

### Added
- Added smoke coverage that keeps the Sample Dataset Guide load options synchronized with the bundled synthetic QA GeoPackage layers.
- Added an explicit loading-modes section to the Sample Dataset Guide HTML report.

## [0.9.4] - 2026-05-22

### Added
- Added `planx_geostats_synthetic_qa.gpkg`, a separate deterministic QA fixture with point, line, polygon, and minimal model-output layers.
- Added synthetic QA sample smoke checks for layer presence, geometry types, feature counts, model-output fields, binary fields, and count fields.
- Added synthetic QA core-engine smoke checks for ANN, Ripley's K, GLR logistic/Poisson, and Linear Directional Mean.
- Added static smoke coverage for multipart guards around direct `asPolyline()` / `asPolygon()` geometry conversions.
- Added a Sample Dataset Guide load selector for the Izmir planning sample, the synthetic QA fixture, or both datasets.

### Changed
- Expanded the Sample Dataset Guide and sample-data README to document both the Izmir planning sample and the synthetic QA fixture.
- Removed several unused runtime imports detected during deep QA cleanup.
- Hardened release code paths so the packaged plugin scans with no Bandit findings in local validation.
- Updated release validation workflow for version `0.9.4`.

## [0.9.3] - 2026-05-21

### Fixed
- Fixed Incremental Spatial Autocorrelation HTML report generation on QGIS 3.40 by avoiding `html` module shadowing.
- Fixed KNN spatial weight generation on QGIS 3.40 by using the current `QgsSpatialIndex.nearestNeighbor` API with a legacy fallback.

### Added
- Added smoke coverage for QGIS spatial-index API compatibility and HTML module shadowing in report writers.

## [0.9.2] - 2026-05-21

### Changed
- Removed developer-only `tests/` files from release zip packages so QGIS Hub security scanning only evaluates runtime plugin code.
- Removed hidden `.gitignore` metadata from release zip packages; it remains in the source repository only.
- Added release-zip verification to the developer validation workflow.

### Fixed
- Resolved the QGIS Hub Bandit block caused by source-only SQLite smoke-test SQL strings being included in the uploaded plugin zip.
- Cleaned several low-risk Hub quality warnings, including unused imports, import ordering, unused local variables, and whitespace style issues.
- Replaced a runtime `assert` in the library installer with an explicit Processing exception.

## [0.9.1] - 2026-05-21

### Added
- Added a QGIS-independent provider catalog smoke test to protect algorithm registration, unique Processing ids, display names, and workflow group coverage.
- Expanded core smoke coverage for GLR family validation and Poisson likelihood accounting.
- Added a plugin-local `.gitignore` for Python caches, IDE files, OS files, and QGIS backup artifacts.
- Added simple, distinct PNG icons for every Processing algorithm and wired each algorithm to its own icon.

### Fixed
- Corrected Poisson GLR log-likelihood and AIC calculation to include the count factorial term.

### Changed
- Kept optional-library status and installation workflows only under `00 | Setup and Diagnostics`; removed the separate GeoStats Libraries menu/toolbar UI.

### Removed
- Removed the unused dependency dialog UI now that setup workflows are Processing-only.

## [0.9.0] - 2026-05-20

### Changed
- Renamed the Processing provider to **PlanX GeoStats Lab**.
- Reorganized tools into English PlanX planning workflow groups: Data Preparation and Neighborhoods, Urban Pattern Scan, Hot Spots and Spatial Outliers, Centers Direction and Dispersion, and Models and Scenarios.
- Moved dependency installation out of Processing and into the built-in GeoStats Libraries helper under the PlanX GeoStats Lab menu.
- Expanded the GeoStats Libraries helper with detailed English guide text, package role explanations, command preview guidance, install-mode guidance, and restart guidance.
- Moved the GeoStats Libraries menu action under **PlanX GeoStats Lab > GeoStats Libraries** so it is scoped to the GeoStats plugin rather than the general PlanX menu.
- Added a GeoStats Libraries toolbar action so the helper remains visible even when QGIS nests plugin menus differently.
- Corrected dependency command generation so QGIS application executables are not shown as pip runners; the helper and status report now distinguish the QGIS host application from the Python executable used for pip.
- Expanded the library status report with a clear "How to install" section for guided installation and manual OSGeo Shell usage.
- Added a Processing Toolbox installer, `Install / Update GeoStats Libraries`, under `00 | Setup and Diagnostics` for profiles where the menu helper is not visible.
- Improved the Processing installer with preview-only behavior, detected path logging, and smarter default mode selection.
- Fixed Global Moran's I z-score and p-value calculation in the shared statistics engine.
- Fixed General G value alignment for QGIS layers whose feature IDs are not contiguous zero-based indices.
- Added neighbor-count diagnostic fields to Gi* and Local Moran output layers.
- Added shared analysis diagnostics for numeric quality, CRS warnings, and neighborhood graph summaries.
- Upgraded Global Moran and General G reports with executive summaries, diagnostics, caveats, and recommended next actions.
- Added OLS model-quality diagnostics for sample size, skipped records, near-constant predictors, multicollinearity, and condition number.
- Enhanced Exploratory Regression reports with model-quality checks, AICc rank reasons, and final-model review guidance.
- Upgraded Average Nearest Neighbor and Incremental Autocorrelation reports with executive summaries, CRS caveats, and next-action guidance.
- Added neighbor-support diagnostics to Incremental Autocorrelation distance scan results.
- Upgraded Sensitivity Test reports with Monte Carlo interpretation, input diagnostics, neighborhood diagnostics, caveats, and next-action guidance.
- Upgraded GWR outputs and reports with local support diagnostics, model-quality checks, CRS caveats, and bandwidth interpretation guidance.
- Added audit fields to Multivariate Clustering and Similarity Search outputs for cluster size, cluster distance, similarity percentile, and similarity tier.
- Added weight-quality checks and audit fields to Mean Center, Standard Distance, and Standard Deviational Ellipse outputs.
- Added Ripley's K-Function as a new urban pattern scan tool with L(d)-d diagnostics, neighborhood support, and planning caveats.
- Added Generalized Linear Regression (GLR) with Gaussian, Logistic, and Poisson model families.
- Added Spatial Autoregression as a PySAL spreg-based spatial lag model with rho diagnostics, neighbor support, output audit fields, and a PlanX analyst report.
- Added Spatial Error Regression (SEM) with lambda diagnostics, residual spatial checks, audit fields, and model interpretation guidance.
- Added Multiscale Geographically Weighted Regression (MGWR) using PySAL mgwr with variable-specific bandwidths, local coefficient audit fields, and a scale-focused analyst report.
- Added Model Comparison Matrix for comparing PlanX model output layers by fit metrics, coverage, bias, and residual spatial autocorrelation.
- Added curated English-schema Izmir neighborhood sample GeoPackage for development, demos, manual QA, and regression workflow testing.
- Added Sample Dataset Guide to load the bundled sample layer from the Processing Toolbox and explain recommended workflows.
- Added Data Readiness Audit under setup diagnostics to review CRS risk, numeric field completeness, constant indicators, sample workflow readiness, and recommended next actions before formal analysis.
- Expanded Data Readiness Audit with geometry QA for empty, invalid, and multipart features before contiguity, local statistics, and distance-based workflows.
- Added a multicollinearity screen to Data Readiness Audit so high-correlation numeric field pairs are flagged before OLS, GLR, GWR, MGWR, and spatial regression workflows.
- Added analysis-role suggestions to Data Readiness Audit so numeric fields are labeled as target/pattern variables, explanatory candidates, count/intensity indicators, or review-only fields with likely tool guidance.
- Added optional field-level CSV export to Data Readiness Audit for spreadsheet QA logs and audit handoffs.
- Expanded Data Readiness Audit workflow guidance with target fields, candidate explanatory fields, recommended tool sequences, and planning-purpose notes for starter analyses.
- Added optional full JSON export to Data Readiness Audit for reproducible audit handoffs, automation, and downstream QA integration.
- Added distribution-shape diagnostics to Data Readiness Audit, including median, quartiles, IQR, skewness, IQR outlier counts, distribution notes, and risk warnings for skewed or outlier-heavy fields.
- Expanded bundled sample-data smoke tests to enforce English schema stability, numeric field types, expected value ranges, complete critical fields, and sufficient variation for analysis workflows.
- Added Bivariate Spatial Association (Lee's L) for local cross-variable neighborhood diagnostics.
- Added shared residual spatial autocorrelation diagnostics to GLR, GWR, Spatial Autoregression, and MGWR reports.
- Added QGIS-independent core smoke tests for the main statistics engines and diagnostics helpers.
- Documented the smoke-test and packaging validation workflow in the README.
- Cleaned remaining special symbols from Processing feedback and generated reports.
- Replaced remaining legacy hyphenated labels in logs and generated HTML reports with PlanX GeoStats Lab.
- Reworked optional type hints to avoid Python 3.10-only union syntax in QGIS 3.28-era Python environments.
- Added an Exploratory Regression safety limit so overly large candidate-variable searches fail fast with guidance instead of tying up QGIS.

### Added
- Registered Central Feature and Incremental Spatial Autocorrelation in the provider.
- Exploratory Regression tool for screening OLS variable combinations and ranking candidate models by AICc.
- Exploratory Regression report now records how many candidate models were estimated.
- GeoStats Library Status diagnostic algorithm under `00 | Setup and Diagnostics`, producing a non-installing HTML dependency report and QGIS Python command preview.
- Install / Update GeoStats Libraries diagnostic algorithm with explicit approval, command logging, OSGeo Shell mode, and restart guidance.

## [0.8.0] - 2026-05-20

### Added
- **Assessing Sensitivity** toolset:
  - Attribute Randomization Sensitivity Test — Monte Carlo permutation simulation for Global Moran's I with SVG histogram HTML report.
- **Measuring Geographic Distributions** toolset:
  - Linear Directional Mean — circular weighted mean orientation for line features with trend line output.

## [0.7.0] - 2026-05-20

### Added
- **Mapping Clusters** toolset:
  - Multivariate Clustering (K-Means) — K-Means++ initialization, Z-score standardization, auto categorized symbology.
- **Utilities** toolset:
  - Export Feature Attributes to CSV/ASCII — configurable delimiter, optional centroid coordinates.

## [0.6.0] - 2026-05-20

### Added
- **Mapping Clusters** toolset:
  - Similarity Search — Z-score attribute profiling with Euclidean/Manhattan distance metrics, auto graduated symbology.
- **Spatial Component Utilities** toolset:
  - Calculate Distance Band from Neighbor Count — k-th neighbor distance statistics with percentile HTML report.

## [0.5.0] - 2026-05-20

### Added
- **Measuring Geographic Distributions** toolset:
  - Median Center — Weiszfeld's algorithm for weighted spatial median.
- **Analyzing Patterns** toolset:
  - High/Low Clustering (Getis-Ord General G) — global G index with randomization variance, HTML report.

## [0.4.0] - 2026-05-20

### Added
- **Analyzing Patterns** toolset:
  - Average Nearest Neighbor (ANN) — chunked KDTree with z-score significance testing, HTML report.
- **Measuring Geographic Distributions** toolset:
  - Standard Distance — weighted circular dispersion polygon at 1/2/3 std dev.
- **Modeling Spatial Relationships** toolset:
  - Geographically Weighted Regression (GWR) — Fixed Gaussian, Fixed/Adaptive Bisquare kernels, local coefficients, auto graduated symbology on local R².

## [0.3.0] - 2026-05-20

### Added
- **Analyzing Patterns** toolset:
  - Global Moran's I — spatial autocorrelation with randomization variance.
- **Modeling Spatial Relationships** toolset:
  - OLS Spatial Regression — ordinary least squares with diagnostics HTML report.

## [0.2.0] - 2026-05-20

### Added
- **Mapping Clusters** toolset:
  - Local Moran's I (LISA) — local spatial autocorrelation with HH/HL/LH/LL classification, auto categorized symbology.
- **Measuring Geographic Distributions** toolset:
  - Mean Center — weighted arithmetic mean center.
  - Standard Deviational Ellipse (SDE) — orientation, semi-axes, ellipse polygon output.
- Dependency Installer utility algorithm.

## [0.1.0] - 2026-05-20

### Added
- **Mapping Clusters** toolset:
  - Getis-Ord Gi* Hot Spot Analysis — z-score and p-value with auto graduated symbology.
- Initial plugin skeleton, metadata, icons, and Processing provider registration.
