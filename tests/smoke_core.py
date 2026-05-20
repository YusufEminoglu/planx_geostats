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


def run_all() -> None:
    test_global_moran_finite_output()
    test_global_moran_zero_variance_is_graceful()
    test_general_g_non_contiguous_ids()
    test_sparse_neighbor_summary()
    test_regression_quality_flags_collinearity()
    test_incremental_autocorrelation_neighbor_diagnostics()
    test_gwr_reports_local_support()
    test_ripleys_k_returns_l_diagnostics()
    print("CORE SMOKE TESTS OK")


if __name__ == "__main__":
    run_all()
