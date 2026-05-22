# -*- coding: utf-8 -*-
"""Pure recommendation logic for the PlanX GeoStats Workflow Advisor."""
from __future__ import annotations


GOAL_OPTIONS = [
    "Explore spatial pattern",
    "Map hot spots or outliers",
    "Choose a distance band",
    "Summarize center, spread, or direction",
    "Build an explanatory model",
    "Compare candidate models",
]
GEOMETRY_OPTIONS = ["Point", "Line", "Polygon or area"]
OUTCOME_OPTIONS = ["No outcome field", "Continuous numeric", "Binary 0/1", "Count"]


def personalized_recommendation(goal: int, geometry: int, outcome: int, has_explanatory: bool) -> dict:
    """Return workflow advice for the selected analytical context."""
    goal_label = _option(GOAL_OPTIONS, goal, 0)
    geometry_label = _option(GEOMETRY_OPTIONS, geometry, 2)
    outcome_label = _option(OUTCOME_OPTIONS, outcome, 1)

    warnings = []
    samples = ["Run Sample Dataset Guide and load the Izmir planning sample for planning workflows."]

    if geometry == 1 and goal not in {3}:
        warnings.append("Line geometry is only directly supported by a subset of GeoStats tools; convert line summaries to points or use Linear Directional Mean when the line direction is the analytical target.")
    if outcome == 0 and goal in {1, 4, 5}:
        warnings.append("This goal usually needs an analysis or dependent field; choose a numeric, binary, or count field before running the recommended sequence.")
    if has_explanatory and goal in {0, 1, 2, 3}:
        warnings.append("Predictor fields are available, but the selected goal is not primarily a modeling workflow; use them later in OLS, GLR, GWR, MGWR, SAR, or SEM.")

    if goal == 2:
        steps = ["Data Readiness Audit", "Calculate Distance Band", "Incremental Spatial Autocorrelation", "Sensitivity Test"]
        summary = "Start with distance-band selection before running local or global spatial statistics."
        samples = ["Izmir: median_land_surface_temp_c", "Synthetic QA: qa_points_grid"]
    elif goal == 3 and geometry == 1:
        steps = ["Data Readiness Audit", "Linear Directional Mean", "Directional Distribution if converted to representative points"]
        summary = "Use line-specific directional tools first, then summarize converted point/centroid patterns if needed."
        samples = ["Synthetic QA: qa_lines_directional"]
    elif goal == 3:
        steps = ["Data Readiness Audit", "Mean Center", "Median Center", "Standard Distance", "Directional Distribution"]
        summary = "Use descriptive geography tools to summarize center, spread, and directional tendency."
        samples = ["Izmir: official_population as optional weight", "Izmir: neighborhood polygons as centroid-derived summaries"]
    elif goal == 4 or has_explanatory:
        if outcome == 2:
            first_model = "Generalized Linear Regression with Logistic family"
            samples = ["Synthetic QA: binary_target with explanatory_a and explanatory_b"]
        elif outcome == 3:
            first_model = "Generalized Linear Regression with Poisson family"
            samples = ["Synthetic QA: count_target with explanatory_a and explanatory_b"]
        else:
            first_model = "OLS Regression"
            samples = ["Izmir: median_land_surface_temp_c with median_ndvi, park_m2_per_capita, tree_canopy_coverage_pct, impervious_surface_pct"]
        steps = ["Data Readiness Audit", first_model, "Residual spatial autocorrelation review", "GWR or MGWR", "Spatial Lag or Spatial Error Regression", "Model Comparison Matrix"]
        summary = f"Build a transparent global baseline for a {outcome_label.lower()} outcome, then compare spatial alternatives."
    elif goal == 5:
        steps = ["Data Readiness Audit", "Run at least two candidate model tools", "Model Comparison Matrix", "Residual map review", "Document final model rationale"]
        summary = "Compare candidate model outputs with fit, coverage, residual spatial pattern, and planning interpretability."
        samples = ["Synthetic QA: qa_ols_model_output, qa_glr_model_output, qa_gwr_model_output, qa_sar_model_output, qa_sem_model_output, qa_mgwr_model_output"]
    elif goal == 1:
        steps = ["Data Readiness Audit", "Calculate Distance Band", "Getis-Ord Gi*", "Local Moran's I", "Bivariate Lee's L if comparing two variables"]
        summary = "Use local statistics to locate hot spots, cold spots, and spatial outliers."
        samples = ["Izmir: median_land_surface_temp_c", "Izmir: senior_65plus_population", "Izmir: median_elevation_m with median_land_surface_temp_c for Lee's L"]
    elif geometry == 0 and outcome == 0:
        steps = ["Data Readiness Audit", "Average Nearest Neighbor", "Ripley's K"]
        summary = "Use point-pattern tools because no analysis field is available."
        samples = ["Synthetic QA: qa_points_grid"]
    else:
        steps = ["Data Readiness Audit", "Calculate Distance Band", "Global Moran's I", "Incremental Spatial Autocorrelation", "Getis-Ord Gi* or Local Moran's I"]
        summary = "Screen the global pattern first, then move to local statistics if spatial structure is present."
        samples = ["Izmir: median_land_surface_temp_c", "Izmir: median_heat_island_index", "Izmir: median_ndvi"]

    checks = [
        f"Geometry context: {geometry_label}.",
        f"Outcome field type: {outcome_label}.",
        "Use a projected CRS for distance-based tools.",
        "Review skipped records, neighborhood support, and residual diagnostics before presenting results.",
    ]
    return {
        "summary": summary,
        "steps": steps,
        "checks": checks,
        "samples": samples,
        "warnings": warnings,
        "goal": goal_label,
    }


def _option(options: list[str], idx: int, default_idx: int) -> str:
    return options[idx] if 0 <= idx < len(options) else options[default_idx]
