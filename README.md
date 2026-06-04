# PlanX GeoStats Lab

PlanX GeoStats Lab is a QGIS Processing provider for spatial statistics in planning workflows. All user-facing tools live inside the Processing Toolbox, including optional-library diagnostics and the explicit library installer under **PlanX GeoStats Lab > 00 | Setup and Diagnostics**.

The `GeoStats Workflow Advisor` tool provides a planning-oriented method guide with user-selectable analysis goals, personalized recommended tool sequences, tool-selection rules, assumptions, common pitfalls, safer moves, and starter recipes for the bundled Izmir and synthetic QA samples.

For release preparation, use `QA_MANUAL_TEST_MATRIX.md` as the manual test checklist covering setup tools, pattern statistics, geometry summaries, modeling workflows, symbology, report interpretation, and release gates.

The main report decision logic is intentionally kept in QGIS-independent core helpers so it can be smoke-tested without launching QGIS. Current guarded helpers cover workflow advising, model-comparison scoring, Monte Carlo sensitivity interpretation, Global Moran's I report interpretation, and Spatial Gini inequality decomposition.

## Sample Data

The plugin includes `sample_data/planx_geostats_izmir_neighborhoods.gpkg`, a compact English-schema GeoPackage with 237 Izmir neighborhood polygons and planning indicators for heat, vegetation, population, parks, street-network structure, building form, and model QA. Use this dataset as the default development and manual testing fixture for PlanX GeoStats workflows. The sample smoke test protects the English schema, critical numeric field types, expected value ranges, and analysis-ready variation. In QGIS, run `PlanX GeoStats Lab > 00 | Setup and Diagnostics > Sample Dataset Guide` to load the layer and open a short workflow guide, then run `Data Readiness Audit` to review geometry validity, field completeness, CRS risk, constant indicators, distribution shape, outlier burden, multicollinearity risk, suggested analysis roles, starter workflow sequences, and recommended analysis paths before launching the statistical tools. The audit can also export a field-level CSV for spreadsheet QA logs and a full JSON package for reproducible audit handoffs.

The plugin also includes `sample_data/planx_geostats_synthetic_qa.gpkg`, a small deterministic QA fixture with point, line, polygon, and minimal model-output layers. It complements the Izmir planning sample by exercising QGIS runtime branches for KNN weights, multipart line handling, model-comparison schemas, report generation, and binary/count model fields. The Sample Dataset Guide can load the Izmir planning sample, the synthetic QA fixture, or both datasets into the current QGIS project.

## Tool Groups

- `00 | Setup and Diagnostics`: library checks, guided optional dependency installation, bundled sample-data loading, workflow advising, and pre-analysis data readiness reports.
- `01 | Data Preparation and Neighborhoods`: tools for preparing attribute exports and choosing neighborhood distance parameters before a statistical workflow begins.
- `02 | Urban Pattern Scan`: global pattern and inequality tools that help planners understand whether a point or polygon distribution is clustered, dispersed, spatially autocorrelated, or spatially unequal across the study area.
- `03 | Hot Spots and Spatial Outliers`: local pattern tools for finding statistically meaningful concentrations, cold spots, cluster/outlier classes, and feature similarity groups.
- `04 | Centers, Direction and Dispersion`: geographic distribution tools for mean/median centers, central features, standard distance, directional ellipses, and linear directional trends.
- `05 | Models and Scenarios`: OLS regression, generalized linear regression, spatial lag/error regression, exploratory regression, GWR, MGWR, model comparison, and sensitivity tools for testing explanatory variables, spatial dependence, multiscale local relationships, and scenario robustness.

## GeoStats Libraries Guide

Open Processing Toolbox and run **PlanX GeoStats Lab > 00 | Setup and Diagnostics > GeoStats Library Status** when an advanced method reports that optional Python libraries are missing. The status tool checks the active QGIS Python environment, shows which packages are available, explains what each package contributes, and previews the exact pip command.

To install from the Toolbox, run **Install / Update GeoStats Libraries**, choose `QGIS Python pip` or `OSGeo Shell`, and review the Processing log. With the confirmation checkbox disabled, the tool prints a preview command and stops; with the checkbox enabled, it runs pip and streams the install log. The installer never runs silently.

QGIS plugins run inside QGIS's own Python process, so installing into a system Python, Anaconda environment, or IDE interpreter will not help unless QGIS is using that same interpreter. The command preview is therefore part of the workflow: it lets the user confirm that the executable path belongs to QGIS before changing the environment.

After installation, restart QGIS completely. Newly installed Python modules may not be visible to already-loaded Processing providers until the application starts a fresh Python process.

## Developer Validation

Run the QGIS-independent smoke tests before packaging:

```powershell
py -3 planx_geostats\tests\smoke_core.py
py -3 planx_geostats\tests\smoke_sample_data.py
py -3 planx_geostats\tests\smoke_provider_catalog.py
py -3 packaging\test_verify_release_zip.py
py -3 packaging\validate_plugin.py planx_geostats --strict
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\Build-PluginZip.ps1 -PluginDir planx_geostats -PluginsRoot C:\Users\YE\PyCharmMiscProject\qgis_plugins
py -3 packaging\verify_release_zip.py QGIS_Plugin_Releases\planx_geostats.zip --root planx_geostats --version 0.9.17
```

The release zip verifier also checks that developer-only paths are absent, algorithm icons are present, metadata points to a packaged icon, and the plugin remains Processing-only without menu or toolbar UI hooks.
