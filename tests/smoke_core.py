# -*- coding: utf-8 -*-
"""QGIS-independent smoke tests for PlanX GeoStats core engines."""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stats = load_module("stats_engines", "core/stats_engines.py")
diagnostics = load_module("analysis_diagnostics", "core/analysis_diagnostics.py")
sys.modules.setdefault("qgis", types.ModuleType("qgis"))
sys.modules.setdefault("qgis.core", types.ModuleType("qgis.core"))
qgis_core = sys.modules["qgis.core"]
for _name in ["QgsFeature", "QgsVectorLayer", "QgsSpatialIndex", "QgsRectangle", "QgsFeedback"]:
    if not hasattr(qgis_core, _name):
        setattr(qgis_core, _name, type(_name, (), {}))
weights_core = load_module("weights_core", "core/weights.py")
workflow_advisor = load_module("workflow_advisor_core", "core/workflow_advisor.py")
model_audit = load_module("model_audit_core", "core/model_audit.py")
sensitivity_audit = load_module("sensitivity_audit_core", "core/sensitivity_audit.py")
spatial_autocorrelation_audit = load_module(
    "spatial_autocorrelation_audit_core",
    "core/spatial_autocorrelation_audit.py",
)
local_pattern_audit = load_module("local_pattern_audit_core", "core/local_pattern_audit.py")
layer_metadata = load_module("layer_metadata_core", "core/layer_metadata.py")
dependencies = load_module("dependencies_core", "dependencies.py")


def assert_finite_tuple(values, expected_len: int) -> None:
    assert len(values) == expected_len
    assert all(np.isfinite(float(value)) for value in values)


def test_global_moran_finite_output() -> None:
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    weights = {0: [1.0], 1: [0.5, 0.5], 2: [0.5, 0.5], 3: [1.0]}
    result = stats.calculate_global_moran(
        np.array([1.0, 2.0, 4.0, 8.0]),
        neighbors,
        weights,
        [0, 1, 2, 3],
    )
    assert_finite_tuple(result, 5)
    assert 0.0 <= result[4] <= 1.0


def test_global_moran_zero_variance_is_graceful() -> None:
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    weights = {0: [1.0], 1: [0.5, 0.5], 2: [0.5, 0.5], 3: [1.0]}
    result = stats.calculate_global_moran(
        np.array([5.0, 5.0, 5.0, 5.0]),
        neighbors,
        weights,
        [0, 1, 2, 3],
    )
    assert result == (0.0, -1.0 / 3.0, 0.0, 0.0, 1.0)


def test_spatial_gini_decomposition_matches_pairwise_definition() -> None:
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    result = stats.calculate_spatial_gini(
        np.array([1.0, 2.0, 4.0, 8.0]),
        neighbors,
        [0, 1, 2, 3],
        permutations=19,
        seed=123,
    )
    assert abs(result["gini"] - (23.0 / 60.0)) < 1e-12
    assert abs(result["neighbor_component"] - (7.0 / 60.0)) < 1e-12
    assert abs(result["non_neighbor_component"] - (16.0 / 60.0)) < 1e-12
    assert abs(result["spatial_gini"] - (16.0 / 23.0)) < 1e-12
    assert abs(result["polarization"] - (16.0 / 7.0)) < 1e-12
    assert result["neighbor_pair_count"] == 3
    assert result["non_neighbor_pair_count"] == 3
    assert 0.0 <= result["p_sim"] <= 1.0


def test_spatial_gini_zero_inequality_is_graceful() -> None:
    result = stats.calculate_spatial_gini(
        np.array([0.0, 0.0, 0.0]),
        {0: [1], 1: [0, 2], 2: [1]},
        [0, 1, 2],
        permutations=9,
    )
    assert result["gini"] == 0.0
    assert result["neighbor_component"] == 0.0
    assert result["non_neighbor_component"] == 0.0
    assert result["spatial_gini"] == 0.0
    assert result["p_sim"] is None


def test_general_g_non_contiguous_ids() -> None:
    neighbors = {10: [20], 20: [10, 50], 50: [20, 90], 90: [50]}
    weights = {10: [1.0], 20: [1.0, 1.0], 50: [1.0, 1.0], 90: [1.0]}
    values = {10: 1.0, 20: 2.0, 50: 4.0, 90: 8.0}
    result = stats.calculate_general_g(values, neighbors, weights, [10, 20, 50, 90])
    assert_finite_tuple(result, 5)
    assert 0.0 <= result[4] <= 1.0


def test_sparse_neighbor_summary() -> None:
    summary = diagnostics.neighbor_summary(
        {10: [], 20: [50], 50: [20], 90: []},
        [10, 20, 50, 90],
    )
    assert summary["minimum"] == 0
    assert summary["maximum"] == 1
    assert summary["isolated"] == 2
    assert summary["density_label"] == "Sparse with isolated observations"


def test_filter_weights_to_valid_ids_restandardizes_rows() -> None:
    neighbors, weights, id_order = diagnostics.filter_weights_to_valid_ids(
        {10: [20, 30], 20: [10, 30, 40], 40: [20], 90: [10]},
        [10, 20, 40],
    )
    assert id_order == [10, 20, 40]
    assert neighbors == {10: [20], 20: [10, 40], 40: [20]}
    assert weights[10] == [1.0]
    assert weights[20] == [0.5, 0.5]
    assert weights[40] == [1.0]


def test_residual_spatial_autocorrelation_summary_flags_pattern() -> None:
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2], 4: [5], 5: [4]}
    weights = {0: [1.0], 1: [0.5, 0.5], 2: [0.5, 0.5], 3: [1.0], 4: [1.0], 5: [1.0]}
    summary = diagnostics.residual_spatial_autocorrelation_summary(
        np.array([3.0, 2.5, 2.0, 1.5, -2.0, -2.5]),
        neighbors,
        weights,
        [0, 1, 2, 3, 4, 5],
    )
    assert summary["available"]
    assert np.isfinite(summary["moran_i"])
    assert 0.0 <= summary["p_value"] <= 1.0
    assert summary["neighbor_summary"]["minimum"] == 1


def test_model_fit_summary_returns_core_metrics() -> None:
    summary = diagnostics.model_fit_summary(
        [10.0, 12.0, 14.0, 16.0],
        [9.0, 13.0, 15.0, 15.0],
        [1.0, -1.0, -1.0, 1.0],
    )
    assert summary["n"] == 4
    assert np.isfinite(summary["r2"])
    assert summary["rmse"] == 1.0
    assert summary["mae"] == 1.0
    assert summary["bias"] == 0.0


def test_regression_quality_flags_collinearity() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
        [5.0, 10.0],
    ])
    summary = diagnostics.regression_quality_summary(y, x, ["x1", "x2"], 6)
    assert summary["used_records"] == 5
    assert summary["skipped_records"] == 1
    assert summary["high_correlations"]
    assert summary["condition_number"] is not None
    assert summary["vif"]
    assert all(len(item) == 2 for item in summary["vif"])


def test_incremental_autocorrelation_neighbor_diagnostics() -> None:
    results = stats.calculate_incremental_autocorrelation(
        np.array([0.0, 1.0, 2.0, 10.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0, 9.0]),
        1.5,
        4.0,
        3,
    )
    assert len(results) == 3
    for result in results:
        assert "min_neighbors" in result
        assert "median_neighbors" in result
        assert "max_neighbors" in result
        assert "isolated_count" in result


def test_gwr_reports_local_support() -> None:
    y = np.array([1.0, 2.0, 2.5, 4.0, 5.0, 6.0])
    x = np.array([[1.0], [2.0], [2.5], [4.0], [5.0], [6.0]])
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
    ])
    result = stats.calculate_gwr(y, x, coords, 4, "adaptive_bisquare")
    assert "local_support" in result
    assert len(result["local_support"]) == len(y)
    assert np.min(result["local_support"]) > 0


def test_ripleys_k_returns_l_diagnostics() -> None:
    results = stats.calculate_ripleys_k(
        np.array([0.0, 1.0, 2.0, 5.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        1.0,
        1.0,
        3,
        study_area=10.0,
    )
    assert len(results) == 3
    for row in results:
        assert "observed_k" in row
        assert "expected_k" in row
        assert "l_minus_d" in row
        assert "isolated_count" in row


def test_glr_families_return_fitted_values() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    gaussian = stats.calculate_glr(np.array([1.0, 2.0, 2.8, 4.2, 5.1, 5.9]), x, "gaussian")
    assert gaussian["converged"]
    assert len(gaussian["fitted"]) == len(x)
    logistic = stats.calculate_glr(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), x, "logistic")
    assert len(logistic["fitted"]) == len(x)
    assert np.all((logistic["fitted"] >= 0.0) & (logistic["fitted"] <= 1.0))
    poisson = stats.calculate_glr(np.array([0.0, 1.0, 1.0, 3.0, 5.0, 8.0]), x, "poisson")
    assert len(poisson["fitted"]) == len(x)
    assert np.all(poisson["fitted"] >= 0.0)
    expected_log_likelihood = float(np.sum(
        np.array([0.0, 1.0, 1.0, 3.0, 5.0, 8.0]) * np.log(poisson["fitted"])
        - poisson["fitted"]
        - np.array([math.lgamma(value + 1.0) for value in [0.0, 1.0, 1.0, 3.0, 5.0, 8.0]])
    ))
    assert abs(poisson["log_likelihood"] - expected_log_likelihood) < 1e-9


def test_glr_rejects_invalid_family_values() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    try:
        stats.calculate_glr(np.array([0.0, 1.0, 1.5, 2.0, 3.0]), x, "logistic")
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("Logistic GLR should reject non-binary dependent values")

    try:
        stats.calculate_glr(np.array([0.0, 1.0, 1.5, 2.0, 3.0]), x, "poisson")
    except ValueError as exc:
        assert "count" in str(exc)
    else:
        raise AssertionError("Poisson GLR should reject non-integer count values")


def test_bivariate_lee_l_returns_classes() -> None:
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    weights = {0: [1.0], 1: [0.5, 0.5], 2: [0.5, 0.5], 3: [1.0]}
    lee_l, lag_y, classes = stats.calculate_bivariate_lee_l(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        neighbors,
        weights,
        [0, 1, 2, 3],
    )
    assert len(lee_l) == 4
    assert len(lag_y) == 4
    assert len(classes) == 4


def test_spatial_index_nearest_neighbor_adapter_prefers_qgis_3_api() -> None:
    class FakeIndex:
        def nearestNeighbor(self, _point, count):
            return list(range(count))

    nearest = weights_core.nearest_neighbor_ids(FakeIndex(), object(), 4)
    assert nearest == [0, 1, 2, 3]


def test_spatial_index_nearest_neighbor_adapter_supports_legacy_plural_api() -> None:
    class FakeIndex:
        def nearestNeighbors(self, _point, count):
            return list(range(count))

    nearest = weights_core.nearest_neighbor_ids(FakeIndex(), object(), 3)
    assert nearest == [0, 1, 2]


def test_geometry_empty_adapter_treats_missing_and_bad_geometry_as_empty() -> None:
    class EmptyGeometry:
        def isEmpty(self):
            return True

    class PresentGeometry:
        def isEmpty(self):
            return False

    class BrokenGeometry:
        def isEmpty(self):
            raise RuntimeError("provider geometry error")

    assert weights_core.geometry_is_missing_or_empty(None)
    assert weights_core.geometry_is_missing_or_empty(EmptyGeometry())
    assert weights_core.geometry_is_missing_or_empty(BrokenGeometry())
    assert not weights_core.geometry_is_missing_or_empty(PresentGeometry())


def test_geometry_centroid_adapter_returns_none_for_bad_centroids() -> None:
    class Point:
        def x(self):
            return 10.0

        def y(self):
            return 20.0

    class Centroid:
        def isEmpty(self):
            return False

        def asPoint(self):
            return Point()

    class Geometry:
        def isEmpty(self):
            return False

        def centroid(self):
            return Centroid()

    class BrokenCentroidGeometry:
        def isEmpty(self):
            return False

        def centroid(self):
            raise RuntimeError("centroid failed")

    point = weights_core.geometry_centroid_point(Geometry())
    assert point.x() == 10.0
    assert point.y() == 20.0
    assert weights_core.geometry_centroid_point(None) is None
    assert weights_core.geometry_centroid_point(BrokenCentroidGeometry()) is None


def test_workflow_advisor_recommends_poisson_glr_for_count_modeling() -> None:
    recommendation = workflow_advisor.personalized_recommendation(goal=4, geometry=2, outcome=3, has_explanatory=True)
    assert "Poisson" in recommendation["steps"][1]
    assert any("count_target" in sample for sample in recommendation["samples"])
    assert not recommendation["warnings"]


def test_workflow_advisor_warns_for_line_hotspot_mismatch() -> None:
    recommendation = workflow_advisor.personalized_recommendation(goal=1, geometry=1, outcome=1, has_explanatory=False)
    assert "Getis-Ord Gi*" in recommendation["steps"]
    assert recommendation["warnings"]
    assert "Line geometry" in recommendation["warnings"][0]


def test_workflow_advisor_recommends_point_pattern_without_field() -> None:
    recommendation = workflow_advisor.personalized_recommendation(goal=0, geometry=0, outcome=0, has_explanatory=False)
    assert recommendation["steps"] == ["Data Readiness Audit", "Average Nearest Neighbor", "Ripley's K"]
    assert any("qa_points_grid" in sample for sample in recommendation["samples"])


def test_model_audit_assigns_sequential_ranks_and_penalizes_residual_pattern() -> None:
    comparisons = [
        {
            "usable": True,
            "layer_name": "low_error_patterned",
            "fit": {"rmse": 1.0, "mae": 1.0},
            "residual_spatial": {"available": True, "p_value": 0.01},
            "coverage": 1.0,
        },
        {
            "usable": True,
            "layer_name": "moderate_error_clean",
            "fit": {"rmse": 1.15, "mae": 1.1},
            "residual_spatial": {"available": True, "p_value": 0.62},
            "coverage": 1.0,
        },
        {
            "usable": True,
            "layer_name": "missing_residual_diagnostic",
            "fit": {"rmse": 1.05, "mae": 1.05},
            "residual_spatial": {"available": False, "p_value": None},
            "coverage": 0.9,
        },
    ]
    model_audit.assign_model_scores(comparisons)
    ranks = {item["layer_name"]: item["rank"] for item in comparisons}
    assert sorted(ranks.values()) == [1, 2, 3]
    assert ranks["moderate_error_clean"] == 1
    assert model_audit.residual_pattern_penalty({"available": True, "p_value": 0.01}) > 0.0
    assert model_audit.residual_pattern_penalty({"available": True, "p_value": 0.20}) == 0.0


def test_model_audit_recommendation_prefers_clean_residual_candidate() -> None:
    comparisons = [
        {
            "usable": True,
            "layer_name": "best_rmse_but_patterned",
            "fit": {"rmse": 1.0, "mae": 1.0},
            "residual_spatial": {"available": True, "p_value": 0.01},
            "coverage": 1.0,
        },
        {
            "usable": True,
            "layer_name": "clean_candidate",
            "fit": {"rmse": 1.2, "mae": 1.1},
            "residual_spatial": {"available": True, "p_value": 0.50},
            "coverage": 1.0,
        },
    ]
    recommendation = model_audit.model_recommendation(comparisons)
    assert "clean_candidate" in recommendation
    assert "without a strong global residual spatial pattern" in recommendation


def test_sensitivity_audit_flags_robust_and_sparse_cases() -> None:
    robust = sensitivity_audit.sensitivity_verdict(
        {"observed_i": 0.42, "empirical_p": 0.01},
        {"isolated": 0, "density_label": "Usable neighborhood graph"},
    )
    assert robust["verdict"].startswith("ROBUST")
    assert "follow-up local analysis" in robust["next_action"]
    assert robust["cautions"] == ["No major automatic sensitivity warning was triggered."]

    sparse = sensitivity_audit.sensitivity_verdict(
        {"observed_i": 0.05, "empirical_p": 0.40},
        {"isolated": 2, "density_label": "Sparse with isolated observations"},
    )
    assert sparse["verdict"].startswith("SENSITIVE")
    assert "Increase the distance band" in sparse["next_action"]
    assert any("isolated observations" in item for item in sparse["cautions"])


def test_global_moran_interpretation_prioritizes_neighbor_graph_risk() -> None:
    clustered = spatial_autocorrelation_audit.global_moran_interpretation(
        3.2,
        0.001,
        {"isolated": 0, "all_connected": False},
    )
    assert clustered["pattern"] == "Clustered"
    assert clustered["confidence"] == "very strong"
    assert "Local Moran" in clustered["next_action"]

    isolated = spatial_autocorrelation_audit.global_moran_interpretation(
        3.2,
        0.001,
        {"isolated": 1, "all_connected": False},
    )
    assert isolated["pattern"] == "Clustered"
    assert "Increase the distance band" in isolated["next_action"]

    dispersed = spatial_autocorrelation_audit.global_moran_interpretation(
        -2.5,
        0.02,
        {"isolated": 0, "all_connected": False},
    )
    assert dispersed["pattern"] == "Dispersed"
    assert dispersed["confidence"] == "strong"


def test_local_pattern_audit_summarizes_hotspot_and_lisa_classes() -> None:
    gi_summary = local_pattern_audit.getis_ord_class_summary([-3, -2, 0, 1, 2, 3, 3])
    assert gi_summary["hot_count"] == 4
    assert gi_summary["cold_count"] == 2
    assert gi_summary["significant_count"] == 6
    assert "dominant class" in gi_summary["message"]

    lisa_summary = local_pattern_audit.local_moran_class_summary(["HH", "HH", "LL", "HL", "LH", "Not Significant"])
    assert lisa_summary["cluster_count"] == 3
    assert lisa_summary["outlier_count"] == 2
    assert lisa_summary["significant_count"] == 5
    assert "Local Moran classified" in lisa_summary["message"]


def test_layer_metadata_survives_alias_unavailable_and_keeps_properties() -> None:
    class Fields:
        def lookupField(self, name):
            return 0 if name == "known" else -1

    class Layer:
        def __init__(self):
            self.properties = {}
            self.aliases = {}

        def setCustomProperty(self, key, value):
            self.properties[key] = value

        def fields(self):
            return Fields()

        def setFieldAlias(self, idx, value):
            if value == "raise":
                raise RuntimeError("alias unavailable")
            self.aliases[idx] = value

    layer = Layer()
    layer_metadata.apply_output_metadata(
        layer,
        "QA layer",
        {"known": "Known field", "missing": "Missing field", "bad_alias": "raise"},
        "qa_algorithm",
    )
    assert layer.properties["planx_geostats:algorithm"] == "qa_algorithm"
    assert layer.properties["planx_geostats:title"] == "QA layer"
    assert layer.properties["planx_geostats:field:known"] == "Known field"
    assert layer.properties["planx_geostats:field:missing"] == "Missing field"
    assert layer.aliases[0] == "Known field"


def test_optional_dependency_error_guides_qgis_toolbox_installation() -> None:
    message = dependencies.optional_dependency_error(
        "Spatial Autoregression",
        ["libpysal", "spreg"],
        ImportError("No module named spreg"),
    )
    required_terms = [
        "Spatial Autoregression",
        "libpysal, spreg",
        "GeoStats Library Status",
        "Install / Update GeoStats Libraries",
        "explicit approval",
        "Restart QGIS",
        "numba",
        "Import error",
    ]
    missing = [term for term in required_terms if term not in message]
    assert not missing, f"Optional dependency guidance is missing: {missing}"


def run_all() -> None:
    test_global_moran_finite_output()
    test_global_moran_zero_variance_is_graceful()
    test_spatial_gini_decomposition_matches_pairwise_definition()
    test_spatial_gini_zero_inequality_is_graceful()
    test_general_g_non_contiguous_ids()
    test_sparse_neighbor_summary()
    test_filter_weights_to_valid_ids_restandardizes_rows()
    test_residual_spatial_autocorrelation_summary_flags_pattern()
    test_model_fit_summary_returns_core_metrics()
    test_regression_quality_flags_collinearity()
    test_incremental_autocorrelation_neighbor_diagnostics()
    test_gwr_reports_local_support()
    test_ripleys_k_returns_l_diagnostics()
    test_glr_families_return_fitted_values()
    test_glr_rejects_invalid_family_values()
    test_bivariate_lee_l_returns_classes()
    test_spatial_index_nearest_neighbor_adapter_prefers_qgis_3_api()
    test_spatial_index_nearest_neighbor_adapter_supports_legacy_plural_api()
    test_geometry_empty_adapter_treats_missing_and_bad_geometry_as_empty()
    test_geometry_centroid_adapter_returns_none_for_bad_centroids()
    test_workflow_advisor_recommends_poisson_glr_for_count_modeling()
    test_workflow_advisor_warns_for_line_hotspot_mismatch()
    test_workflow_advisor_recommends_point_pattern_without_field()
    test_model_audit_assigns_sequential_ranks_and_penalizes_residual_pattern()
    test_model_audit_recommendation_prefers_clean_residual_candidate()
    test_sensitivity_audit_flags_robust_and_sparse_cases()
    test_global_moran_interpretation_prioritizes_neighbor_graph_risk()
    test_local_pattern_audit_summarizes_hotspot_and_lisa_classes()
    test_layer_metadata_survives_alias_unavailable_and_keeps_properties()
    test_optional_dependency_error_guides_qgis_toolbox_installation()
    print("CORE SMOKE TESTS OK")


if __name__ == "__main__":
    run_all()
