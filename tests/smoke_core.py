# -*- coding: utf-8 -*-
"""QGIS-independent smoke tests for PlanX GeoStats core engines."""
from __future__ import annotations

import importlib.util
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


def run_all() -> None:
    test_global_moran_finite_output()
    test_global_moran_zero_variance_is_graceful()
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
    test_bivariate_lee_l_returns_classes()
    print("CORE SMOKE TESTS OK")


if __name__ == "__main__":
    run_all()
