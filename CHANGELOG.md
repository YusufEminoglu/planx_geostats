# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
