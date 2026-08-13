# PlanX GeoStats Lab Sample Data

This folder contains curated sample data for developing, demonstrating, and manually testing PlanX GeoStats Lab workflows.

## Dataset

`planx_geostats_izmir_fur.gpkg`

- Layer name: `izmir_fur_street_network`
- Geometry: multipolygon
- CRS: EPSG:5253 (TUREF / TM27)
- Feature count: 391 polygons covering the Izmir Functional Urban Region (FUR)
- Purpose: street-network and space-syntax accessibility sample data for spatial statistics, hot spot analysis, local outlier diagnostics, regression, spatial regression, GWR/MGWR, model comparison, and report QA.

All 34 attribute fields are numeric - there is no name, id, or categorical field in this export. Reference individual polygons by `fid` in example workflows. The source data is provided directly by the plugin author; the original export is not modified beyond renaming the file and layer for the plugin's bundling convention.

## Suggested Analysis Fields

- `road_density`: the default "intensity" field for pattern-scan and hot-spot examples throughout the manual and Workflow Advisor - non-negative, well varied (313 distinct values), and shows strong, genuinely significant clustering (Global Moran's I = 0.63, z > 27 at KNN=8).
- `betweenness_mean`, `ss_integration_median`: network-centrality and space-syntax fields used for the hot-spot/local-outlier family (Getis-Ord Gi*, Local Moran's I, Bivariate Lee's L).
- `transit_accessibility`: the default dependent/outcome field for regression examples (OLS, GWR, MGWR, Spatial Lag, Spatial Error), paired with `road_density`, `gridiron_index`, `connectivity_index`, and `circuity_mean` as explanatory fields.
- `gridiron_index`, `connectivity_index`, `circuity_mean`: street-pattern-structure fields used as explanatory variables in the model examples.
- `pedestrian_ratio`: constant at 0.0 across every feature in this export (no pedestrian-only streets recorded) - a deliberate, real "zero-variation field" example for Data Readiness Audit's constant-field detection.

## Field Dictionary

All fields below are `REAL` type. Roughly 4 of 391 features (~1%) are null across most fields, and ~75 features (~19%) have a fully zero-filled network profile - both are legitimate characteristics of peripheral/undeveloped FUR zones, not data defects, and make good Data Readiness Audit teaching examples.

| Field | Description |
| --- | --- |
| `road_density` | Total road length per unit area within the zone. |
| `intersection_count` | Count of street intersections within the zone. |
| `intersection_density` | Intersections per unit area. |
| `dead_end_ratio` | Share of street segments ending in a dead end / cul-de-sac. |
| `lane_count_median` | Median number of travel lanes across streets in the zone. |
| `street_width` | Median representative street width. |
| `block_length_median` | Median urban block length. |
| `arterial_ratio` | Share of arterial (higher-order) road length relative to total road length. |
| `pedestrian_ratio` | Share of pedestrian-only street length. Constant at 0.0 in this export. |
| `circuity_mean` | Mean ratio of network distance to straight-line distance (detour factor); 1.0 is a perfectly direct network. |
| `orientation_entropy` | Shannon entropy of street-segment bearing distribution; low values indicate a strongly gridded/aligned pattern, high values a disordered orientation. The only field with zero missing values across all 391 features. |
| `gridiron_index` | Index of grid-like regularity of the local street pattern. |
| `reach_500m_road_length_km`, `reach_500m_road_length` | Total road length reachable within a 500 m network buffer (km and native units). |
| `reach_800m_road_length`, `reach_800m_intersection` | Total road length / intersection count reachable within an 800 m network buffer. |
| `transit_accessibility` | Composite accessibility score to transit service. |
| `emergency_corridor_criticality` | Criticality score of the zone's roads as emergency-response corridors. |
| `ss_choice_median` | Median space-syntax "choice" value (betweenness-like through-movement potential). |
| `ss_integration_median` | Median space-syntax "integration" value (closeness-like to-movement potential). |
| `nach_median`, `nach_index` | Normalized Angular Choice (NACH) - scale-independent choice. |
| `nain_median`, `nain_index` | Normalized Angular Integration (NAIN) - scale-independent integration. |
| `node_degree_mean` | Mean node degree of the street-network graph (average links per junction). |
| `betweenness_mean` | Mean node betweenness centrality. |
| `eigenvector_mean` | Mean node eigenvector centrality. |
| `clustering_mean` | Mean local clustering coefficient of the network graph. |
| `bridge_node_ratio` | Share of nodes that are graph bridge (cut) nodes. |
| `n_betweenness`, `n_eigenvector` | Normalized (0-1 scaled) betweenness / eigenvector centrality. |
| `connectivity_index` | Composite street-connectivity index. |
| `link_criticality` | Criticality score of network links - how much connectivity would be lost if the link were removed. |
| `street_facing_continuity` | Continuity of street-facing building frontage. |
| `SHAPE_Length`, `SHAPE_Area` | Polygon perimeter/area fields carried over from the original ArcGIS export; redundant with QGIS's own `$length`/`$area` expressions and kept as-is from the source data. |

## Notes for Development

- Keep this sample small enough for normal Processing tests and manual QA. If a new algorithm is computationally expensive, use a subset or a temporary filtered layer during development.
- Keep all sample-facing field names, report examples, and documentation in English.
- Prefer this dataset when testing PlanX GeoStats report language, diagnostics, and model-comparison workflows.

## Synthetic QA Fixture

`planx_geostats_synthetic_qa.gpkg`

- CRS: EPSG:3857
- Purpose: compact runtime QA fixture for geometry types and output schemas not covered by the Izmir FUR polygon sample.
- Layers:
  - `qa_points_grid`: 25 point features with continuous, binary, and count fields for ANN, Ripley's K, distance-band, autocorrelation, GLR, and regression smoke checks.
  - `qa_lines_directional`: 6 line/multiline features for Linear Directional Mean and multipart line handling.
  - `qa_polygons_mini`: 9 compact polygons for queen/rook contiguity and small local-statistics checks.
  - `qa_ols_model_output`, `qa_glr_model_output`, `qa_gwr_model_output`, `qa_sar_model_output`, `qa_sem_model_output`, `qa_mgwr_model_output`: minimal model-output layers for Model Comparison Matrix detection and report QA.

Keep this fixture deterministic and intentionally small. It is not intended to represent a real planning geography; it exists to exercise QGIS Processing runtime branches, API compatibility, geometry handling, and report generation.
