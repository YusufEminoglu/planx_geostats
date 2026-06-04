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
                    "median_land_surface_temp_c",
                    "median_ndvi",
                    "building_coverage_pct",
                    "normalized_integration",
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
                "FIELD": "median_land_surface_temp_c",
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
                "FIELD": "park_m2_per_capita",
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
            "getis_ord_gi",
            "Hot Spot Analysis (Getis-Ord Gi*)",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD": "median_land_surface_temp_c",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "local_moran_lisa",
            "Local Moran's I",
            lambda env: {
                "INPUT": env["izmir"],
                "FIELD": "median_land_surface_temp_c",
                "WEIGHT_TYPE": 2,
                "KNN": 8,
                "DISTANCE_BAND": 1000.0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "bivariate_spatial_association_lees_l",
            "Bivariate Spatial Association (Lee's L)",
            lambda env: {
                "INPUT": env["izmir"],
                "X_FIELD": "median_elevation_m",
                "Y_FIELD": "median_land_surface_temp_c",
                "WEIGHT_TYPE": 2,
                "KNN": 5,
                "DISTANCE_BAND": 1000.0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
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
        ),
        RuntimeCase(
            "mean_center",
            "Mean Center",
            lambda env: {
                "INPUT": env["izmir"],
                "WEIGHT_FIELD": "official_population",
                "MODE": 0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "central_feature",
            "Central Feature",
            lambda env: {
                "INPUT": env["izmir"],
                "WEIGHT_FIELD": "official_population",
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "median_center",
            "Median Center",
            lambda env: {
                "INPUT": env["izmir"],
                "WEIGHT_FIELD": "official_population",
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "standard_distance",
            "Standard Distance",
            lambda env: {
                "INPUT": env["izmir"],
                "WEIGHT_FIELD": "official_population",
                "MULTIPLIER": 0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "directional_distribution",
            "Directional Distribution",
            lambda env: {
                "INPUT": env["izmir"],
                "WEIGHT_FIELD": "official_population",
                "STD_DEV": 0,
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "linear_directional_mean",
            "Linear Directional Mean",
            lambda env: {
                "INPUT": env["qa_lines"],
                "OUTPUT": "memory:",
            },
            layer_outputs={"OUTPUT": 1},
        ),
        RuntimeCase(
            "ols_regression",
            "Ordinary Least Squares Regression",
            lambda env: {
                "INPUT": env["izmir"],
                "DEP_VAR": "median_land_surface_temp_c",
                "INDEPENDENTS": ["median_ndvi", "building_coverage_pct", "normalized_integration"],
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_ols_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
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
            optional_dependency_ok=True,
        ),
        RuntimeCase(
            "exploratory_regression",
            "Exploratory Regression",
            lambda env: {
                "INPUT": env["izmir"],
                "DEPENDENT_FIELD": "median_land_surface_temp_c",
                "EXPLANATORY_FIELDS": [
                    "median_ndvi",
                    "building_coverage_pct",
                    "normalized_integration",
                    "impervious_area_m2",
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
        ),
        RuntimeCase(
            "multiscale_geographically_weighted_regression",
            "Multiscale Geographically Weighted Regression",
            lambda env: {
                "INPUT": env["qa_points"],
                "DEP_VAR": "target_value",
                "INDEPENDENTS": ["explanatory_a", "explanatory_b"],
                "KERNEL_TYPE": 0,
                "CRITERION": 0,
                "MIN_BW": 6.0,
                "MAX_BW": 12.0,
                "MAX_ITER": 3,
                "N_CHUNKS": 1,
                "SPHERICAL": False,
                "OUTPUT": "memory:",
                "HTML_REPORT": _path(env["out"], "05_mgwr_regression.html"),
            },
            html_outputs=("HTML_REPORT",),
            layer_outputs={"OUTPUT": 1},
            optional_dependency_ok=True,
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
    sample = plugin / "sample_data" / "planx_geostats_izmir_neighborhoods.gpkg"
    synthetic = plugin / "sample_data" / "planx_geostats_synthetic_qa.gpkg"

    env = {
        "root": root,
        "plugin": plugin,
        "out": out_dir,
        "izmir": _load_layer(sample, "planx_geostats_izmir_neighborhoods", "PlanX QA Izmir"),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="qgis_plugins root")
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
