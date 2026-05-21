# PlanX GeoStats Lab Sample Data

This folder contains curated sample data for developing, demonstrating, and manually testing PlanX GeoStats Lab workflows.

## Dataset

`planx_geostats_izmir_neighborhoods.gpkg`

- Layer name: `planx_geostats_izmir_neighborhoods`
- Geometry: polygon
- CRS: EPSG:5253
- Feature count: 237 neighborhood polygons
- Purpose: compact urban-planning sample data for spatial statistics, hot spot analysis, local outlier diagnostics, regression, spatial regression, GWR/MGWR, model comparison, and report QA.

The source data was provided as `planxgeostats_sample.gpkg` and curated into an English, analysis-friendly schema. The original source file is not modified. Column names in this sample use stable snake_case names so Processing models, scripts, and tests can reference them safely.

## Suggested Analysis Fields

- `median_heat_island_index`: good dependent variable for heat exposure, Global Moran, Gi*, Local Moran, OLS, GWR, MGWR, SAR, and SEM examples.
- `median_land_surface_temp_c`: continuous temperature surface metric for pattern scanning and regression.
- `median_ndvi`: vegetation index for green-infrastructure and cooling relationships.
- `building_coverage_pct`: built-form intensity indicator.
- `impervious_area_m2`: impervious-surface exposure indicator.
- `park_m2_per_capita`: public green-space access indicator; contains nulls where park data is not available.
- `street_connectivity`, `normalized_choice`, `normalized_integration`: street-network structure indicators.
- `official_population`, `child_under5_population`, `senior_65plus_population`, `female_population`: population and vulnerability indicators.

## Field Dictionary

| Field | Description |
| --- | --- |
| `neighborhood_name` | Neighborhood name. |
| `district_name` | District name. |
| `neighborhood_code` | Neighborhood code from the source data. |
| `official_population` | Official population count. |
| `raster_population` | Population estimate from raster/zonal aggregation. |
| `female_population` | Female population estimate. |
| `child_under5_population` | Population under age 5. |
| `senior_65plus_population` | Population age 65 and older. |
| `closeness_centrality` | Median street-network closeness centrality. |
| `betweenness_centrality` | Median street-network betweenness centrality. |
| `street_connectivity` | Median street-network connectivity; use as an ordinal network-support indicator rather than a highly continuous model variable. |
| `median_segment_length_m` | Median street segment length in meters. |
| `normalized_choice` | Normalized angular choice indicator. |
| `normalized_integration` | Normalized angular integration indicator. |
| `median_elevation_m` | Median elevation in meters. |
| `median_slope_deg` | Median slope in degrees. |
| `median_canopy_height_m` | Median canopy height in meters. |
| `canopy_area_m2` | Tree canopy area in square meters. |
| `median_built_up_index` | Median built-up index. |
| `median_water_index` | Median water index. |
| `median_ndvi` | Median normalized difference vegetation index. |
| `median_savi` | Median soil-adjusted vegetation index. |
| `median_land_surface_temp_c` | Median land surface temperature in Celsius. |
| `median_heat_island_index` | Median urban heat island indicator. |
| `median_thermal_field_index` | Median UTFWI-style thermal field indicator. |
| `impervious_area_m2` | Impervious surface area in square meters. |
| `emergency_facility_count` | Count of emergency/disaster-related facilities. |
| `park_count` | Count of parks intersecting or assigned to the neighborhood. |
| `park_area_m2` | Total park area in square meters. |
| `park_m2_per_capita` | Park area per capita. |
| `neighborhood_area_m2` | Neighborhood polygon area in square meters. |
| `building_footprint_area_m2` | Building footprint area in square meters. |
| `building_coverage_pct` | Source-derived building coverage intensity indicator. Values are generally percentage-like but can exceed 100 in the source sample; review this field in Data Readiness Audit before formal modeling. |
| `urban_density_class` | Source density class: Low, Moderate, High, or Very High. |
| `median_building_height_m` | Median building height in meters. |
| `building_volume_density_pct` | Combined 2D/3D building density percentage. |

## Notes for Development

- Keep this sample small enough for normal Processing tests and manual QA. If a new algorithm is computationally expensive, use a subset or a temporary filtered layer during development.
- Keep all sample-facing field names, report examples, and documentation in English.
- Prefer this dataset when testing PlanX GeoStats report language, diagnostics, and model-comparison workflows.

## Synthetic QA Fixture

`planx_geostats_synthetic_qa.gpkg`

- CRS: EPSG:3857
- Purpose: compact runtime QA fixture for geometry types and output schemas not covered by the Izmir polygon planning sample.
- Layers:
  - `qa_points_grid`: 25 point features with continuous, binary, and count fields for ANN, Ripley's K, distance-band, autocorrelation, GLR, and regression smoke checks.
  - `qa_lines_directional`: 6 line/multiline features for Linear Directional Mean and multipart line handling.
  - `qa_polygons_mini`: 9 compact polygons for queen/rook contiguity and small local-statistics checks.
  - `qa_ols_model_output`, `qa_glr_model_output`, `qa_gwr_model_output`, `qa_sar_model_output`, `qa_sem_model_output`, `qa_mgwr_model_output`: minimal model-output layers for Model Comparison Matrix detection and report QA.

Keep this fixture deterministic and intentionally small. It is not intended to represent a real planning geography; it exists to exercise QGIS Processing runtime branches, API compatibility, geometry handling, and report generation.
