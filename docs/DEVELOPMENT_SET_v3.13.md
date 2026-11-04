# GeoStats Lab 3.13.0 — development set

Written 2026-11-04 by the master engineer from two independent audits of
3.12.1 (GS-CODE: numerics and robustness against esda and hand formulas;
GS-UX: first use, legibility, truth, demand), both run offscreen on QGIS
3.44.12 and 4.2.0 with fresh profiles. Evidence: `gsx/gs-code` and
`gsx/gs-ux` in the session scratchpad.

3.12.1 gets the core right: queen/rook/KNN weights, global and local Moran,
Geary C and Gi* match esda, permutations are seeded, optional dependencies fail
cleanly, the Hub scan is clean. What hurts users is a wrong General G
variance, results that read as "Random" when nothing was computed, and maps
that hide or mislabel features. Demand: plain, correct Gi*/LISA with FDR and a
projected-CRS guard (Hotspot Analysis v3 got 6.4k downloads since May 2026).

## 1. Problems
| # | Harm | Where |
|---|------|-------|
| P1 | General G variance ~7× too large (z understated ~2.7×): real clustering reported as random. | `core/stats_engines.py:1100-1101` |
| P2 | No neighbours (S0 = 0, e.g. points with the default queen weights) reports "I 0, p 1, Random"; Local Moran/Gi* all "Not Significant". | `stats_engines.py:476` |
| P3 | NaN attribute values pass `float()`: Moran nan "Random", Local Moran all nan "Not Significant", Gi* z = 0, Geary crashes in `charts.py:421`. | `alg_local_moran.py:222`, `alg_getis_ord.py:241`, `alg_geary_c.py:210`, … |
| P4 | OLS p-values use the normal distribution, not Student t (n = 12: p 0.028 vs exact 0.055). | `stats_engines.py:359` |
| P5 | Weights not re-standardised after NULL rows are dropped (I_i off by up to 0.147) in 12 tools. | `filter_weights_to_valid_ids` not called |
| P6 | Gi* gives an island (no real neighbours) a z from its own value: a far-away polygon became "Hot Spot 99 %", in every multiple-testing mode; help says z = 0, p = 1. | `stats_engines.py:47-49`, `alg_getis_ord.py:102` |
| P7 | Geary with no neighbours: traceback instead of a message. | `advanced_stats_engines.py:87` |
| P8 | Distance band on EPSG:4326 is in degrees (default 1000 = everything neighbours everything); 15 tools never warn. Stability variants scale the band whatever weights were chosen. | `crs_unit_warning` unused |
| P9 | General G is O(n²) memory/time (N = 20k → GBs). | `stats_engines.py:1058,1082`, `alg_general_g.py:207` |
| P10 | Gi (non-star) formula wrong (unused). OLS residual Moran misses n/S0. Cancel during writing returns a partial layer. | |
| P11 | NULL features: Local Moran writes them "Not Significant" (292 vs the report's 288); Gi* leaves gi_conf NULL and the renderer hides them. | `alg_local_moran.py:314` |
| P12 | Dock double-click applies a preset that turns stability on; the map is then styled by the ±2 consensus but the legend says "95 % Confidence". | `geostats_dock.py:57`, `alg_getis_ord.py:302` |
| P13 | Dock tree text 1.36:1 under Night Mapping. | `geostats_dock.py` QSS ~102 |
| P14 | Reports: "fdr_bh" code instead of words, counts "28.00", help says lisa_p where the code uses lisa_padj; workflow messages show internal ids. | |
| P15 | "hotspot" finds nothing in the toolbox or dock search; Hub tags miss hot spot, spatial autocorrelation, local moran, geoda, pysal, esda, spatial weights, regionalization, max-p, skater. | |
| P16 | Truth: README "90 algorithms" (102), 15 sibling links to this repo, a Turkish section, badge/link mismatch; tracker 404; guide link 404; tooltip says GitHub Pages. | README, metadata |
| P17 | Tests: 7 of 9 harmful mutations survive (legacy engines only checked for finite values). | |

## 2. Delivers
1. Correct numbers (P1, P4, P6, P9, P10) checked against esda values.
2. No silent "Random": runs without neighbours or with only NaN fail with a
   plain message; NaN rows are skipped and counted; weights rebuilt on valid
   rows; contiguity weights on points refused with a hint (KNN or distance
   band); a distance band on a geographic CRS refused with a hint (P2, P3,
   P5, P7, P8).
3. Maps that show every feature: a "No data" and a "No neighbours" class,
   counts in the report; a separate "Stable hot/cold" legend for consensus
   (P6, P11, P12).
4. Legible dock, readable reports, findable tools (P13-P15).
5. Honest listing and docs (P16); honest tests (P17).

## 3. Contracts
- C1 (GS-NUM): engines raise `core.stats_engines.InputError` (a ValueError
  whose text is a user sentence) for S0 = 0, all-island weights and
  zero variance; Gi* returns NaN z and p for features with no neighbours
  besides themselves.
- C2 (GS-ALG): algorithms convert InputError to QgsProcessingException;
  output layers use class labels "No data" (value missing) and "No
  neighbours" (island), and Gi* writes `gi_conf = 0` for both.

## 4. Implementers
| Agent | Owns |
|-------|------|
| GS-NUM | `core/stats_engines.py`, `core/advanced_stats_engines.py`, `core/local_stats_engines.py`, `core/weights.py`, `tests/esda_parity.py`, `tests/smoke_core.py`, new `tests/test_numeric_truth.py` |
| GS-ALG | `algorithms/**`, `core/analysis_diagnostics.py`, `core/reporting.py`, `core/symbology.py`, `core/charts.py`, `core/spatial_autocorrelation_audit.py`, `core/local_pattern_audit.py`, new `tests/test_robustness_qgis.py`, `tests/qgis_runtime_algorithm_matrix.py` |
| GS-UI | `geostats_dock.py`, `main_plugin.py`, `planx_geostats_provider.py`, `core/workflow_advisor.py`, new `tests/test_dock_qgis.py` |
| GS-DOCS | `README.md`, `CHANGELOG.md`, `CITATION.cff`, `metadata.txt`, `docs/**` except this file, `tools/manual_src/**` |

## 5. Rules
Offscreen; every QGIS run with a fresh `QGIS_CUSTOM_CONFIG_PATH`; zero
public network traffic; QGIS 3.28 / Python 3.9 and both runtimes; Hub scan
rules; red before green; no git writes; no pf verify.

## 6. Status
- 2026-11-04: audits done; set written; implementers launched.
