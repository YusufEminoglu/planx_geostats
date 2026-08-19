#!/usr/bin/env python3
r"""Run every PlanX GeoStats Processing algorithm against bundled QA data.

Execute with a QGIS Python runtime, for example:
  C:\OSGeo4W\bin\python-qgis-ltr.bat planx_geostats\tests\qgis_runtime_algorithm_matrix.py --root C:\Users\YE\PyCharmMiscProject\qgis_plugins
  C:\OSGeo4W\bin\python-qgis.bat planx_geostats\tests\qgis_runtime_algorithm_matrix.py --root C:\Users\YE\PyCharmMiscProject\qgis_plugins
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


PROVIDER_ID = "planx_geostats"
OPTIONAL_DEPENDENCY_TEXT = "requires optional Python package(s)"
INSTALL_PREVIEW_TEXT = "Preview only: installation was not started"
# TabPFN needs a one-time license acceptance (a cached TABPFN_TOKEN) before its
# first local inference call; a test machine that has never completed that
# out-of-band step is an environment-configuration gap, not a plugin bug -
# tolerate it the same way a genuinely missing optional package is tolerated.
TABPFN_AUTH_TEXT = "one-time license"


@dataclass
class RuntimeCase:
    algorithm: str
    label: str
    params: Callable[[dict], dict]
    html_outputs: tuple[str, ...] = ()
    file_outputs: tuple[str, ...] = ()
    layer_outputs: dict[str, int] = field(default_factory=dict)
    expected_exception: str | None = None
    optional_dependency_ok: bool = False
    # Regression guard for postProcessAlgorithm's layer-styling/metadata step:
    # layer_outputs only checks feature count, which stays green even when
    # postProcessAlgorithm silently no-ops (e.g. the QgsProject.instance()
    # .mapLayer() vs context.getMapLayer() bug fixed in v3.7.0, where the
    # layer lookup returned None and every apply_renderer()/
    # apply_output_metadata() call after it was skipped). renderer_class, if
    # set, asserts the OUTPUT layer's renderer is exactly that QGIS class
    # (e.g. "QgsCategorizedSymbolRenderer"); alias_check_field, if set,
    # asserts that field has a non-empty alias (proof apply_output_metadata
    # actually ran) - used for tools with metadata but no custom renderer.
    renderer_class: str | None = None
    alias_check_field: str | None = None


def _add_root_to_path(root: Path) -> None:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _add_qgis_plugin_path() -> None:
    try:
        import qgis
    except Exception:
        return
    plugin_dir = Path(qgis.__file__).resolve().parent.parent / "plugins"
    if plugin_dir.exists():
        plugin_text = str(plugin_dir)
        if plugin_text not in sys.path:
            sys.path.insert(0, plugin_text)


def _new_qgis_app():
    from qgis.core import QgsApplication
    from qgis.PyQt.QtWidgets import QApplication

    app = QApplication.instance()
    created = False
    if app is None or not isinstance(app, QgsApplication):
        app = QgsApplication([], True)
        created = True
    app.initQgis()
    return app, created


def _cleanup_qgis_app(app, created: bool) -> None:
    if not created:
        return
    if os.name == "nt":
        # OSGeo4W can return a nonzero process code during QGIS teardown after
        # many temporary Processing layers even when every algorithm passed.
        return
    try:
        app.exitQgis()
    except Exception:
        pass


def _init_processing() -> None:
    from processing.core.Processing import Processing

    Processing.initialize()


class CaptureFeedback:
    """Small proxy around QgsProcessingFeedback that keeps useful log tails."""

    def __init__(self):
        from qgis.core import QgsProcessingFeedback

        class _Feedback(QgsProcessingFeedback):
            def __init__(self, outer):
                super().__init__()
                self._outer = outer

            def pushInfo(self, message):
                self._outer.messages.append(("info", str(message)))
                super().pushInfo(message)

            def pushWarning(self, message):
                self._outer.messages.append(("warning", str(message)))
                super().pushWarning(message)

            def reportError(self, message, fatalError=False):
                self._outer.messages.append(("error", str(message)))
                super().reportError(message, fatalError)

        self.messages: list[tuple[str, str]] = []
        self.feedback = _Feedback(self)

    def tail(self, count: int = 8) -> list[str]:
        return [f"{kind}: {text}" for kind, text in self.messages[-count:]]


def _load_layer(gpkg: Path, layer_name: str, display_name: str | None = None):
    from qgis.core import QgsVectorLayer

    uri = f"{gpkg}|layername={layer_name}"
    layer = QgsVectorLayer(uri, display_name or layer_name, "ogr")
    if not layer.isValid():
        raise AssertionError(f"Could not load sample layer: {uri}")
    return layer


def _path(out_dir: Path, name: str) -> str:
    return str(out_dir / name)


def _case_id(algorithm: str) -> str:
    return f"{PROVIDER_ID}:{algorithm}"


def _all_cases() -> list[RuntimeCase]:
    return [
        RuntimeCase(
            "geostats_library_status",
            "GeoStats Library Status",
            lambda env: {"HTML_REPORT": _path(env["out"], "00_library_status.html")},
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "install_geostats_libraries",
            "Install / Update GeoStats Libraries preview",
            lambda env: {"INSTALL_MODE": 0, "CONFIRM": False},
            expected_exception=INSTALL_PREVIEW_TEXT,
        ),
        RuntimeCase(
            "sample_dataset_guide",
            "Sample Dataset Guide",
            lambda env: {
                "LOAD_IN_PROJECT": True,
                "DATASET_TO_LOAD": 2,
                "HTML_REPORT": _path(env["out"], "00_sample_dataset_guide.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "data_readiness_audit",
            "Data Readiness Audit",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELDS": [
                    "road_density",
                    "betweenness_mean",
                    "ss_integration_median",
                    "gridiron_index",
                ],
                "HTML_REPORT": _path(env["out"], "00_data_readiness.html"),
                "FIELD_AUDIT_CSV": _path(env["out"], "00_data_readiness_fields.csv"),
                "AUDIT_JSON": _path(env["out"], "00_data_readiness.json"),
            },
            html_outputs=("HTML_REPORT",),
            file_outputs=("FIELD_AUDIT_CSV", "AUDIT_JSON"),
        ),
        RuntimeCase(
            "geostats_workflow_advisor",
            "GeoStats Workflow Advisor",
            lambda env: {
                "GOAL": 1,
                "GEOMETRY_CONTEXT": 0,
                "OUTCOME_TYPE": 0,
                "HAS_EXPLANATORY": True,
                "HTML_REPORT": _path(env["out"], "00_workflow_advisor.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "calculate_distance_band",
            "Calculate Distance Band",
            lambda env: {
                "INPUT": env["qa_points"],
                "NEIGHBOR_COUNT": 5,
                "HTML_REPORT": _path(env["out"], "01_distance_band.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "export_attributes_to_ascii",
            "Export Feature Attributes to CSV/ASCII",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELDS": ["target_value", "explanatory_a", "explanatory_b", "binary_target"],
                "DELIMITER": 0,
                "INCLUDE_COORDS": True,
                "OUTPUT_FILE": _path(env["out"], "01_export_attributes.csv"),
            },
            file_outputs=("OUTPUT_FILE",),
        ),
        RuntimeCase(
            "global_moran_autocorrelation",
            "Global Moran's I",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD": "road_density",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "HTML_REPORT": _path(env["out"], "02_global_moran.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "spatial_gini_inequality",
            "Spatial Inequality (Gini and Spatial Gini)",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD": "road_density",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "PERMUTATIONS": 19,
                "RANDOM_SEED": 42,
                "HTML_REPORT": _path(env["out"], "02_spatial_gini.html"),
                "SUMMARY_CSV": _path(env["out"], "02_spatial_gini.csv"),
                "SUMMARY_JSON": _path(env["out"], "02_spatial_gini.json"),
            },
            html_outputs=("HTML_REPORT",),
            file_outputs=("SUMMARY_CSV", "SUMMARY_JSON"),
        ),
        RuntimeCase(
            "general_g_autocorrelation",
            "Getis-Ord General G",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELD": "count_target",
                "DISTANCE_BAND": 500.0,
                "HTML_REPORT": _path(env["out"], "02_general_g.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "incremental_spatial_autocorrelation",
            "Incremental Spatial Autocorrelation",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELD": "target_value",
                "START_DISTANCE": 250.0,
                "DISTANCE_INCREMENT": 250.0,
                "N_INCREMENTS": 3,
                "HTML_REPORT": _path(env["out"], "02_incremental_autocorrelation.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "ripleys_k_function",
            "Ripley's K-Function",
            lambda env: {
                "INPUT": env["qa_points"],
                "START_DISTANCE": 250.0,
                "DISTANCE_INCREMENT": 250.0,
                "N_INCREMENTS": 3,
                "STUDY_AREA": 1_000_000.0,
                "HTML_REPORT": _path(env["out"], "02_ripleys_k.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "average_nearest_neighbor",
            "Average Nearest Neighbor",
            lambda env: {
                "INPUT": env["qa_points"],
                "STUDY_AREA": 1_000_000.0,
                "HTML_REPORT": _path(env["out"], "02_average_nearest_neighbor.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "geary_c_autocorrelation",
            "Geary's C",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD": "road_density",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "PERMUTATIONS": 99,
                "RANDOM_SEED": 42,
                "HTML_REPORT": _path(env["out"], "02_geary_c.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "join_count_statistics",
            "Join Count Statistics",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD": "transit_accessibility",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "PERMUTATIONS": 99,
                "RANDOM_SEED": 42,
                "HTML_REPORT": _path(env["out"], "02_join_count.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "global_bivariate_lee_l",
            "Global Bivariate Lee's L",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD_X": "road_density",
                "FIELD_Y": "betweenness_mean",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "PERMUTATIONS": 99,
                "RANDOM_SEED": 42,
                "HTML_REPORT": _path(env["out"], "02_global_lee_l.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "geodetector_q_statistic",
            "Geodetector Q-Statistic",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD_Y": "road_density",
                "STRATA_MODE": 1,
                "FIELD_STRATA": "ss_integration_median",
                "N_BINS": 5,
                "PERMUTATIONS": 99,
                "RANDOM_SEED": 42,
                "HTML_REPORT": _path(env["out"], "02_geodetector_q.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "getis_ord_gi",
            "Hot Spot Analysis (Getis-Ord Gi*)",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD": "betweenness_mean",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "03_getis_ord.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsCategorizedSymbolRenderer",
        ),
        RuntimeCase(
            "local_moran_lisa",
            "Local Moran's I",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD": "ss_integration_median",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "03_local_moran.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsCategorizedSymbolRenderer",
        ),
        RuntimeCase(
            "bivariate_lisa",
            "Bivariate LISA",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD_X": "road_density",
                "FIELD_Y": "betweenness_mean",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "PERMUTATIONS": 99,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "03_bivariate_lisa.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsCategorizedSymbolRenderer",
        ),
        RuntimeCase(
            "bivariate_spatial_association_lees_l",
            "Bivariate Spatial Association (Lee's L)",
            lambda env: {
                "INPUT": env["izmir"],
                "X_FIELD": "road_density",
                "Y_FIELD": "betweenness_mean",
                "WEIGHT_TYPE": 2,
                "KNN": 5,
                "DISTANCE_BAND": 1000.0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsCategorizedSymbolRenderer",
        ),
        RuntimeCase(
            "multivariate_clustering",
            "Multivariate Clustering",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELDS": ["target_value", "explanatory_a", "explanatory_b"],
                "K_CLUSTERS": 3,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsCategorizedSymbolRenderer",
        ),
        RuntimeCase(
            "similarity_search",
            "Similarity Search",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELDS": ["explanatory_a", "explanatory_b"],
                "TARGET_EXPRESSION": '"fid" = 1',
                "METRIC": 0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsGraduatedSymbolRenderer",
        ),
        RuntimeCase(
            "local_geary_c",
            "Local Geary's C",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD": "road_density",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "PERMUTATIONS": 99,
                "RANDOM_SEED": 42,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "03_local_geary_c.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsCategorizedSymbolRenderer",
        ),
        RuntimeCase(
            "colocation_quotient",
            "Colocation Quotient",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELD": "cluster_hint",
                "CATEGORY_A": "core",
                "CATEGORY_B": "edge",
                "KNN": 3,
                "PERMUTATIONS": 99,
                "RANDOM_SEED": 42,
                "HTML_REPORT": _path(env["out"], "03_colocation_quotient.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "skater_regionalization",
            "SKATER Spatially Constrained Regionalization",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELDS": ["road_density", "betweenness_mean", "gridiron_index"],
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "K_CLUSTERS": 3,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsCategorizedSymbolRenderer",
        ),
        RuntimeCase(
            "mean_center",
            "Mean Center",
            lambda env: {
                "INPUT": env["izmir"],
                "WEIGHT_FIELD": "road_density",
                "MODE": 0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
            alias_check_field="mean_x",
        ),
        RuntimeCase(
            "central_feature",
            "Central Feature",
            lambda env: {
                "INPUT": env["izmir"],
                "WEIGHT_FIELD": "road_density",
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
            alias_check_field="is_central",
        ),
        RuntimeCase(
            "median_center",
            "Median Center",
            lambda env: {
                "INPUT": env["izmir"],
                "WEIGHT_FIELD": "road_density",
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
            alias_check_field="median_x",
        ),
        RuntimeCase(
            "standard_distance",
            "Standard Distance",
            lambda env: {
                "INPUT": env["izmir"],
                "WEIGHT_FIELD": "road_density",
                "MULTIPLIER": 0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
            alias_check_field="mean_x",
        ),
        RuntimeCase(
            "directional_distribution",
            "Directional Distribution",
            lambda env: {
                "INPUT": env["izmir"],
                "WEIGHT_FIELD": "road_density",
                "STD_DEV": 0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
            alias_check_field="mean_x",
        ),
        RuntimeCase(
            "linear_directional_mean",
            "Linear Directional Mean",
            lambda env: {
                "INPUT": env["qa_lines"],
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
            alias_check_field="mean_angle",
        ),
        RuntimeCase(
            "ols_regression",
            "Ordinary Least Squares Regression",
            lambda env: {
                "INPUT": env["izmir"],
                "DEP_VAR": "transit_accessibility",
                "INDEPENDENTS": ["road_density", "gridiron_index", "connectivity_index"],
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_ols_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsGraduatedSymbolRenderer",
        ),
        RuntimeCase(
            "lm_diagnostics",
            "Lagrange Multiplier Diagnostics",
            lambda env: {
                "INPUT": env["izmir"],
                "DEP_VAR": "transit_accessibility",
                "INDEPENDENTS": ["road_density", "gridiron_index", "connectivity_index"],
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "HTML_REPORT": _path(env["out"], "05_lm_diagnostics.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "generalized_linear_regression",
            "Generalized Linear Regression",
            lambda env: {
                "INPUT": env["qa_points"],
                "DEP_VAR": "binary_target",
                "INDEPENDENTS": ["explanatory_a", "explanatory_b"],
                "FAMILY": 1,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_glr_logistic.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsGraduatedSymbolRenderer",
        ),
        RuntimeCase(
            "spatial_autoregression",
            "Spatial Autoregression",
            lambda env: {
                "INPUT": env["qa_points"],
                "DEP_VAR": "target_value",
                "INDEPENDENTS": ["explanatory_a", "explanatory_b"],
                "WEIGHT_TYPE": 2,
                "KNN": 4,
                "DISTANCE_BAND": 500.0,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_spatial_autoregression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsGraduatedSymbolRenderer",
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "spatial_error_regression",
            "Spatial Error Regression",
            lambda env: {
                "INPUT": env["qa_points"],
                "DEP_VAR": "target_value",
                "INDEPENDENTS": ["explanatory_a", "explanatory_b"],
                "WEIGHT_TYPE": 2,
                "KNN": 4,
                "DISTANCE_BAND": 500.0,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_spatial_error_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsGraduatedSymbolRenderer",
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "spatial_durbin_model",
            "Spatial Durbin Model",
            lambda env: {
                "INPUT": env["izmir"],
                "DEP_VAR": "transit_accessibility",
                "INDEPENDENTS": ["road_density", "gridiron_index"],
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_spatial_durbin.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsGraduatedSymbolRenderer",
        ),
        RuntimeCase(
            "exploratory_regression",
            "Exploratory Regression",
            lambda env: {
                "INPUT": env["izmir"],
                "DEPENDENT_FIELD": "transit_accessibility",
                "EXPLANATORY_FIELDS": [
                    "road_density",
                    "gridiron_index",
                    "connectivity_index",
                    "circuity_mean",
                ],
                "MAX_VARIABLES": 2,
                "HTML_REPORT": _path(env["out"], "05_exploratory_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "gwr_regression",
            "Geographically Weighted Regression",
            lambda env: {
                "INPUT": env["qa_points"],
                "DEP_VAR": "target_value",
                "INDEPENDENTS": ["explanatory_a", "explanatory_b"],
                "KERNEL_TYPE": 2,
                "BANDWIDTH": 8.0,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_gwr_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsGraduatedSymbolRenderer",
        ),
        RuntimeCase(
            "multiscale_geographically_weighted_regression",
            "Multiscale Geographically Weighted Regression",
            lambda env: {
                "INPUT": env["qa_points"],
                "DEP_VAR": "target_value",
                "INDEPENDENTS": ["explanatory_a", "explanatory_b"],
                # Fixed kernel, not Adaptive: mgwr's multiscale backfitting seeds
                # itself with an internal, unconfigurable "40 + 2*n_vars" minimum
                # record count for adaptive kernels (a mgwr library limitation, see
                # alg_mgwr.py's precondition check), which this 25-record QA fixture
                # is too small for. Fixed kernels use distance-based bounds instead
                # and have no such minimum, so this exercises the real MGWR fit/output
                # path on the small QA fixture rather than only its error handling.
                "KERNEL_TYPE": 3,
                "CRITERION": 0,
                "MIN_BW": 0.0,
                "MAX_BW": 0.0,
                "MAX_ITER": 3,
                "N_CHUNKS": 1,
                "SPHERICAL": False,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_mgwr_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsGraduatedSymbolRenderer",
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "esf_regression",
            "Eigenvector Spatial Filtering Regression",
            lambda env: {
                "INPUT": env["izmir"],
                "DEP_VAR": "transit_accessibility",
                "INDEPENDENTS": ["road_density", "gridiron_index"],
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "MAX_EIGENVECTORS": 8,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_esf_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            renderer_class="QgsGraduatedSymbolRenderer",
        ),
        RuntimeCase(
            "quantile_regression",
            "Quantile Regression",
            lambda env: {
                "INPUT": env["qa_points"],
                "DEP_VAR": "target_value",
                "INDEPENDENTS": ["explanatory_a", "explanatory_b"],
                "TAU": 0.5,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_quantile_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "spatial_regime_regression",
            "Spatial Regime Regression",
            lambda env: {
                "INPUT": env["qa_points"],
                "DEP_VAR": "target_value",
                "INDEPENDENTS": ["explanatory_a", "explanatory_b"],
                "REGIME_FIELD": "cluster_hint",
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_spatial_regime_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "gw_summary_statistics",
            "Geographically Weighted Summary Statistics",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELD": "target_value",
                "KERNEL_TYPE": 2,
                "BANDWIDTH": 8.0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "sensitivity_test",
            "Attribute Randomization Sensitivity Test",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELD": "target_value",
                "DISTANCE_BAND": 500.0,
                "SIMULATIONS": 99,
                "HTML_REPORT": _path(env["out"], "05_sensitivity_test.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "model_comparison_matrix",
            "Model Comparison Matrix",
            lambda env: {
                "MODEL_LAYERS": [
                    env["qa_ols_model_output"],
                    env["qa_glr_model_output"],
                    env["qa_gwr_model_output"],
                    env["qa_sar_model_output"],
                    env["qa_sem_model_output"],
                    env["qa_mgwr_model_output"],
                ],
                "DEP_VAR": "observed_y",
                "HTML_REPORT": _path(env["out"], "05_model_comparison.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        # -------------------------------------------------------------- #
        # 06 | Machine Learning and Explainable AI - previously had ZERO
        # runtime coverage in this file (34 algorithms, including all 8
        # added alongside kNNDM in 2.1.0) even though every other group was
        # exercised here. optional_dependency_ok=True marks the tools that
        # need one of the 8 packages the "GeoStats Library Status" tool can
        # report missing (xgboost, lightgbm, catboost, shap, interpret,
        # mapie, dice-ml, tabpfn); everything else here only needs
        # scikit-learn, which ships with this QGIS runtime already.
        # -------------------------------------------------------------- #
        RuntimeCase(
            "random_forest_regression",
            "Random Forest Regression",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "N_ESTIMATORS": 50,
                "MAX_DEPTH": 0,
                "MIN_SAMPLES_LEAF": 1,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_random_forest_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "random_forest_classification",
            "Random Forest Classification",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "N_ESTIMATORS": 50,
                "MAX_DEPTH": 0,
                "MIN_SAMPLES_LEAF": 1,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_random_forest_classification.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "extra_trees_regression",
            "Extra Trees Regression",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "N_ESTIMATORS": 50,
                "MAX_DEPTH": 0,
                "MIN_SAMPLES_LEAF": 1,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_extra_trees_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "extra_trees_classification",
            "Extra Trees Classification",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "N_ESTIMATORS": 50,
                "MAX_DEPTH": 0,
                "MIN_SAMPLES_LEAF": 1,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_extra_trees_classification.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "support_vector_regression",
            "Support Vector Regression (SVR)",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "KERNEL": 1,
                "C": 1.0,
                "EPSILON": 0.1,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_svr.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "support_vector_classification",
            "Support Vector Classification (SVC)",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "KERNEL": 1,
                "C": 1.0,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_svc.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "gbm_regression_sklearn",
            "Gradient Boosting Regression (scikit-learn)",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "N_ESTIMATORS": 50,
                "LEARNING_RATE": 0.1,
                "MAX_DEPTH": 3,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_gbm_regression_sklearn.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "gbm_regression_xgboost",
            "Gradient Boosting Regression (XGBoost)",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "N_ESTIMATORS": 50,
                "LEARNING_RATE": 0.1,
                "MAX_DEPTH": 3,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_gbm_regression_xgboost.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "gbm_regression_lightgbm",
            "Gradient Boosting Regression (LightGBM)",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "N_ESTIMATORS": 50,
                "LEARNING_RATE": 0.1,
                "MAX_DEPTH": 3,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_gbm_regression_lightgbm.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "gbm_regression_catboost",
            "Gradient Boosting Regression (CatBoost)",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "N_ESTIMATORS": 50,
                "LEARNING_RATE": 0.1,
                "MAX_DEPTH": 3,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_gbm_regression_catboost.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "gbm_classification_sklearn",
            "Gradient Boosting Classification (scikit-learn)",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "N_ESTIMATORS": 50,
                "LEARNING_RATE": 0.1,
                "MAX_DEPTH": 3,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_gbm_classification_sklearn.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "gbm_classification_xgboost",
            "Gradient Boosting Classification (XGBoost)",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "N_ESTIMATORS": 50,
                "LEARNING_RATE": 0.1,
                "MAX_DEPTH": 3,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_gbm_classification_xgboost.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "gbm_classification_lightgbm",
            "Gradient Boosting Classification (LightGBM)",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "N_ESTIMATORS": 50,
                "LEARNING_RATE": 0.1,
                "MAX_DEPTH": 3,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_gbm_classification_lightgbm.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "gbm_classification_catboost",
            "Gradient Boosting Classification (CatBoost)",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "N_ESTIMATORS": 50,
                "LEARNING_RATE": 0.1,
                "MAX_DEPTH": 3,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_gbm_classification_catboost.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "mlp_regression",
            "Neural Network Regression (MLP)",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "HIDDEN_LAYERS": "16,8",
                "MAX_ITER": 500,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_mlp_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "mlp_classification",
            "Neural Network Classification (MLP)",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "HIDDEN_LAYERS": "16,8",
                "MAX_ITER": 500,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_mlp_classification.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "spatial_kfold_cv_evaluator",
            "Spatial k-Fold Cross-Validation Evaluator (kNNDM)",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "target_value",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "TASK_TYPE": 0,
                "MODEL": 0,
                "N_FOLDS": 3,
                "FOLD_METHOD": 1,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_spatial_cv_knndm.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "permutation_feature_importance",
            "Permutation Feature Importance",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "TASK_TYPE": 0,
                "MODEL": 0,
                "N_REPEATS": 5,
                "HTML_REPORT": _path(env["out"], "06_permutation_importance.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "partial_dependence_report",
            "Partial Dependence Report",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "PD_FIELD": "road_density",
                "TASK_TYPE": 0,
                "MODEL": 0,
                "GRID_POINTS": 10,
                "HTML_REPORT": _path(env["out"], "06_partial_dependence.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "ml_model_comparison",
            "ML Model Comparison (Leaderboard)",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "TASK_TYPE": 1,
                "MODELS": [0, 1, 2, 3, 4],
                "N_FOLDS": 3,
                "HTML_REPORT": _path(env["out"], "06_ml_model_comparison.html"),
            },
            html_outputs=("HTML_REPORT",),
        ),
        RuntimeCase(
            "shap_global_importance",
            "SHAP Global Feature Importance",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "TASK_TYPE": 0,
                "MODEL": 0,
                "MAX_ROWS": 100,
                "HTML_REPORT": _path(env["out"], "06_shap_global.html"),
            },
            html_outputs=("HTML_REPORT",),
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "shap_spatial_map",
            "SHAP Spatial Attribution Map",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "TASK_TYPE": 0,
                "MODEL": 0,
                "MAX_ROWS": 100,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_shap_spatial_map.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "shap_local_explanation",
            "SHAP Local Explanation Report",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "TASK_TYPE": 0,
                "MODEL": 0,
                "FEATURE_ID": 1,
                "HTML_REPORT": _path(env["out"], "06_shap_local.html"),
            },
            html_outputs=("HTML_REPORT",),
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "dice_counterfactual_explanation",
            "DiCE Counterfactual Explanation",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "MODEL": 0,
                "FEATURE_ID": 1,
                "DESIRED_CLASS": "",
                "N_COUNTERFACTUALS": 2,
                "IMMUTABLE_FEATURES": [],
                "HTML_REPORT": _path(env["out"], "06_dice_counterfactual.html"),
            },
            html_outputs=("HTML_REPORT",),
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "ebm_regression",
            "Explainable Boosting Machine Regression",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_ebm_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "ebm_classification",
            "Explainable Boosting Machine Classification",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_ebm_classification.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "model_residual_autocorrelation",
            "Model Residual Spatial Autocorrelation Check",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "MODEL": 0,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_model_residual_autocorrelation.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "prediction_uncertainty_map",
            "Prediction Uncertainty Map",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "MODEL": 0,
                "N_ESTIMATORS": 50,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_prediction_uncertainty.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "conformal_prediction_interval",
            "Conformal Prediction Interval",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "MODEL": 0,
                "ALPHA": 0.1,
                "N_FOLDS": 5,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_conformal_prediction.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "tabpfn_regression",
            "TabPFN Regression",
            lambda env: {
                "INPUT": env["izmir"],
                "TARGET": "transit_accessibility",
                "FEATURES": ["road_density", "gridiron_index", "connectivity_index"],
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_tabpfn_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "tabpfn_classification",
            "TabPFN Classification",
            lambda env: {
                "INPUT": env["qa_points"],
                "TARGET": "binary_target",
                "FEATURES": ["explanatory_a", "explanatory_b"],
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "06_tabpfn_classification.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "dbscan_clustering",
            "DBSCAN Density-Based Clustering",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELDS": ["target_value", "explanatory_a", "explanatory_b"],
                "EPS": 1.5,
                "MIN_SAMPLES": 3,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "hdbscan_clustering",
            "HDBSCAN Hierarchical Density Clustering",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELDS": ["target_value", "explanatory_a", "explanatory_b"],
                "MIN_CLUSTER_SIZE": 3,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "gmm_clustering",
            "Gaussian Mixture Model Clustering",
            lambda env: {
                "INPUT": env["qa_points"],
                "FIELDS": ["target_value", "explanatory_a", "explanatory_b"],
                "N_COMPONENTS": 2,
                "COVARIANCE_TYPE": 3,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
    ]


def _register_provider():
    from qgis.core import QgsApplication
    from planx_geostats.planx_geostats_provider import PlanXGeoStatsProvider

    registry = QgsApplication.processingRegistry()
    for provider in list(registry.providers()):
        if provider.id() == PROVIDER_ID:
            registry.removeProvider(provider)
    provider = PlanXGeoStatsProvider()
    if not registry.addProvider(provider):
        raise AssertionError("Could not register PlanX GeoStats provider")
    return provider


def _validate_catalog(provider, cases: list[RuntimeCase]) -> None:
    ids = {alg.id() for alg in provider.algorithms()}
    expected = {_case_id(case.algorithm) for case in cases}
    missing = sorted(expected - ids)
    if missing:
        raise AssertionError(f"Provider is missing algorithm(s): {missing}")
    if len(ids) < len(expected):
        raise AssertionError(f"Provider algorithm count too small: {len(ids)} < {len(expected)}")


def _resolve_layer(value, context):
    if hasattr(value, "isValid") and hasattr(value, "featureCount"):
        return value
    if not value:
        return None
    try:
        from qgis.core import QgsProcessingUtils

        return QgsProcessingUtils.mapLayerFromString(str(value), context)
    except Exception:
        return None


def _verify_file(path_text: str, label: str) -> None:
    path = Path(path_text)
    if not path.exists():
        raise AssertionError(f"{label} was not created: {path}")
    if path.stat().st_size <= 0:
        raise AssertionError(f"{label} is empty: {path}")


def _verify_outputs(case: RuntimeCase, result: dict, context) -> None:
    for key in case.html_outputs:
        path_text = result.get(key) or result.get(f"{key}_OUT")
        if not path_text:
            raise AssertionError(f"{case.algorithm} did not return HTML output key {key}")
        _verify_file(str(path_text), key)

    for key in case.file_outputs:
        path_text = result.get(key) or result.get(f"{key}_OUT")
        if not path_text:
            raise AssertionError(f"{case.algorithm} did not return file output key {key}")
        _verify_file(str(path_text), key)

    for key, min_features in case.layer_outputs.items():
        if key not in result:
            raise AssertionError(f"{case.algorithm} did not return layer output key {key}")
        layer = _resolve_layer(result.get(key), context)
        if layer is None or not layer.isValid():
            raise AssertionError(f"{case.algorithm} returned an invalid layer for {key}: {result.get(key)!r}")
        if int(layer.featureCount()) < min_features:
            raise AssertionError(
                f"{case.algorithm} output {key} has too few features: "
                f"{layer.featureCount()} < {min_features}"
            )
        if case.renderer_class is not None:
            actual = type(layer.renderer()).__name__
            if actual != case.renderer_class:
                raise AssertionError(
                    f"{case.algorithm} output {key} has renderer {actual}, "
                    f"expected {case.renderer_class} - postProcessAlgorithm's "
                    f"styling step likely silently no-op'd (layer lookup failed)"
                )
        if case.alias_check_field is not None:
            idx = layer.fields().lookupField(case.alias_check_field)
            if idx < 0:
                raise AssertionError(f"{case.algorithm} output {key} is missing field {case.alias_check_field}")
            if not layer.fields().at(idx).alias():
                raise AssertionError(
                    f"{case.algorithm} output {key} field {case.alias_check_field} has no alias - "
                    f"postProcessAlgorithm's apply_output_metadata() likely silently no-op'd "
                    f"(layer lookup failed)"
                )


def _run_case(case: RuntimeCase, env: dict, context) -> dict:
    import processing
    from qgis.core import QgsProcessingException

    feedback_capture = CaptureFeedback()
    params = case.params(env)
    try:
        result = processing.run(
            _case_id(case.algorithm),
            params,
            context=context,
            feedback=feedback_capture.feedback,
        )
        if case.expected_exception:
            raise AssertionError(
                f"Expected exception containing {case.expected_exception!r}, "
                "but the algorithm completed successfully"
            )
        _verify_outputs(case, result, context)
        return {
            "algorithm": case.algorithm,
            "label": case.label,
            "ok": True,
            "status": "ok",
            "messages": feedback_capture.tail(),
        }
    except QgsProcessingException as exc:
        message = str(exc)
        if case.expected_exception and case.expected_exception in message:
            return {
                "algorithm": case.algorithm,
                "label": case.label,
                "ok": True,
                "status": "expected_exception",
                "error": message,
                "messages": feedback_capture.tail(),
            }
        if case.optional_dependency_ok and OPTIONAL_DEPENDENCY_TEXT in message:
            return {
                "algorithm": case.algorithm,
                "label": case.label,
                "ok": True,
                "status": "optional_dependency_missing",
                "error": message,
                "messages": feedback_capture.tail(),
            }
        if case.optional_dependency_ok and TABPFN_AUTH_TEXT in message:
            return {
                "algorithm": case.algorithm,
                "label": case.label,
                "ok": True,
                "status": "tabpfn_license_not_accepted",
                "error": message,
                "messages": feedback_capture.tail(),
            }
        return {
            "algorithm": case.algorithm,
            "label": case.label,
            "ok": False,
            "status": "failed",
            "error": message,
            "messages": feedback_capture.tail(),
        }
    except Exception:
        return {
            "algorithm": case.algorithm,
            "label": case.label,
            "ok": False,
            "status": "failed",
            "error": traceback.format_exc(limit=8),
            "messages": feedback_capture.tail(),
        }


def _build_environment(root: Path, out_dir: Path) -> dict:
    plugin = root / "planx_geostats"
    sample = plugin / "sample_data" / "planx_geostats_izmir_fur.gpkg"
    synthetic = plugin / "sample_data" / "planx_geostats_synthetic_qa.gpkg"

    env = {
        "root": root,
        "plugin": plugin,
        "out": out_dir,
        "izmir": _load_layer(sample, "izmir_fur_street_network", "PlanX QA Izmir FUR"),
        "qa_points": _load_layer(synthetic, "qa_points_grid", "PlanX QA Points"),
        "qa_lines": _load_layer(synthetic, "qa_lines_directional", "PlanX QA Lines"),
        "qa_polygons": _load_layer(synthetic, "qa_polygons_mini", "PlanX QA Polygons"),
    }
    for name in (
        "qa_ols_model_output",
        "qa_glr_model_output",
        "qa_gwr_model_output",
        "qa_sar_model_output",
        "qa_sem_model_output",
        "qa_mgwr_model_output",
    ):
        env[name] = _load_layer(synthetic, name, f"PlanX QA {name}")
    return env


def _run_matrix(root: Path, runtime: str, only: set[str], out_dir: Path) -> dict:
    from qgis.core import Qgis, QgsApplication, QgsProcessingContext, QgsProject

    _init_processing()
    provider = _register_provider()
    cases = _all_cases()
    if only:
        cases = [case for case in cases if case.algorithm in only or _case_id(case.algorithm) in only]
    _validate_catalog(provider, _all_cases())

    context = QgsProcessingContext()
    context.setProject(QgsProject.instance())
    env = _build_environment(root, out_dir)

    results = []
    try:
        for case in cases:
            results.append(_run_case(case, env, context))
    finally:
        try:
            context.temporaryLayerStore().removeAllMapLayers()
        except Exception:
            pass
        try:
            QgsProject.instance().removeAllMapLayers()
        except Exception:
            pass
        try:
            QgsApplication.processingRegistry().removeProvider(provider)
        except Exception:
            pass

    ok_count = sum(1 for result in results if result["ok"])
    return {
        "runtime": runtime,
        "qgis_version": getattr(Qgis, "QGIS_VERSION", ""),
        "qgis_version_int": int(getattr(Qgis, "QGIS_VERSION_INT", 0)),
        "output_dir": str(out_dir),
        "case_count": len(results),
        "ok_count": ok_count,
        "failed_count": len(results) - ok_count,
        "results": results,
        "ok": all(result["ok"] for result in results),
    }


def main() -> int:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root, help="qgis_plugins root")
    parser.add_argument("--runtime", default="unknown")
    parser.add_argument("--only", nargs="*", default=[], help="Algorithm ids or names to run")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--keep-outputs", action="store_true")
    args = parser.parse_args()

    if os.name == "nt":
        os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    root = args.root.resolve()
    _add_root_to_path(root)
    _add_qgis_plugin_path()

    app, created = _new_qgis_app()
    try:
        if args.output_dir is not None:
            out_dir = args.output_dir.resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            result = _run_matrix(root, args.runtime, set(args.only), out_dir)
        elif args.keep_outputs:
            out_dir = Path(tempfile.mkdtemp(prefix="planx_geostats_matrix_"))
            result = _run_matrix(root, args.runtime, set(args.only), out_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="planx_geostats_matrix_") as tmp:
                result = _run_matrix(root, args.runtime, set(args.only), Path(tmp))
        print("GEOSTATS_RUNTIME_MATRIX_JSON=" + json.dumps(result, ensure_ascii=True))
        if result["ok"]:
            print(f"GEOSTATS_RUNTIME_MATRIX: PASS ({result['ok_count']}/{result['case_count']})")
            return 0
        print(f"GEOSTATS_RUNTIME_MATRIX: FAIL ({result['failed_count']} failed)")
        for item in result["results"]:
            if not item["ok"]:
                print(f"- {item['algorithm']}: {item.get('error', 'unknown error')}")
        return 1
    finally:
        _cleanup_qgis_app(app, created)


if __name__ == "__main__":
    sys.exit(main())
