# PlanX GeoStats Lab

PlanX GeoStats Lab is a QGIS Processing provider for spatial statistics in planning workflows. It keeps the analytical tools inside the Processing Toolbox, while the built-in **PlanX GeoStats Lab > GeoStats Libraries** helper and toolbar action manage optional Python libraries used by advanced workflows.

## Tool Groups

- `01 | Data Preparation and Neighborhoods`: tools for preparing attribute exports and choosing neighborhood distance parameters before a statistical workflow begins.
- `02 | Urban Pattern Scan`: global pattern tools that help planners understand whether a point or polygon distribution is clustered, dispersed, or spatially autocorrelated across the study area.
- `03 | Hot Spots and Spatial Outliers`: local pattern tools for finding statistically meaningful concentrations, cold spots, cluster/outlier classes, and feature similarity groups.
- `04 | Centers, Direction and Dispersion`: geographic distribution tools for mean/median centers, central features, standard distance, directional ellipses, and linear directional trends.
- `05 | Models and Scenarios`: OLS regression, exploratory regression, geographically weighted modeling, and sensitivity tools for testing explanatory variables and scenario robustness.

## GeoStats Libraries Guide

Open **PlanX GeoStats Lab > GeoStats Libraries** from the QGIS menu when an advanced method reports that optional Python libraries are missing. The helper checks the active QGIS Python environment, shows which packages are available, explains what each package contributes, previews the exact pip command, and runs installation only after explicit confirmation.

The helper intentionally does not install packages silently. QGIS plugins run inside QGIS's own Python process, so installing into a system Python, Anaconda environment, or IDE interpreter will not help unless QGIS is using that same interpreter. The command preview is therefore part of the workflow: it lets the user confirm that the executable path belongs to QGIS before changing the environment.

After installation, restart QGIS completely. Newly installed Python modules may not be visible to already-loaded Processing providers until the application starts a fresh Python process.

If the menu action is not visible in a QGIS profile, open Processing Toolbox and use `PlanX GeoStats Lab > 00 | Setup and Diagnostics > GeoStats Library Status` to inspect the detected Python paths. To install from the Toolbox, run `Install / Update GeoStats Libraries`, choose `QGIS Python pip` or `OSGeo Shell`, enable the confirmation checkbox, and watch the Processing log. The installer never runs silently; it prints the exact command before executing pip.
