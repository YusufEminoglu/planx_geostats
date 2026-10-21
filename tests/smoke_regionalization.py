# -*- coding: utf-8 -*-
"""QGIS-independent tests for regionalisation and partition quality (v3.12.0).

The fixture is a lattice whose attributes are generated from four known
blocks, so every method has a right answer to recover; the remaining tests
pin the constraints every solution must satisfy (contiguity, the max-p floor,
k regions, minimum size) and closed-form values of the quality measures.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_package():
    package = types.ModuleType("gs_core")
    package.__path__ = [str(ROOT / "core")]
    sys.modules["gs_core"] = package
    loaded = {}
    for name in ("local_stats_engines", "regionalization_engines", "region_quality"):
        spec = importlib.util.spec_from_file_location(f"gs_core.{name}", ROOT / "core" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"gs_core.{name}"] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


_M = _load_package()
lse, reg, rq = _M["local_stats_engines"], _M["regionalization_engines"], _M["region_quality"]


def lattice(rows, cols):
    nb, w, ids = {}, {}, []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            ids.append(i)
            ring = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    ring.append(rr * cols + cc)
            nb[i], w[i] = ring, [1.0] * len(ring)
    return lse.SpatialGraph.from_dicts(nb, w, ids)


SIDE = 12
G = lattice(SIDE, SIDE)
ADJ = reg.adjacency_lists(G)
COL = np.tile(np.arange(SIDE), SIDE)
ROW = np.repeat(np.arange(SIDE), SIDE)
TRUTH = (COL >= 6).astype(int) + 2 * (ROW >= 6)
X = np.column_stack([TRUTH * 3.0 + np.random.default_rng(0).normal(scale=0.3, size=SIDE * SIDE),
                     (TRUTH % 2) * 2.0 + np.random.default_rng(1).normal(scale=0.3, size=SIDE * SIDE)])


def _matches_truth(labels):
    # identical partition up to relabelling
    pairs = set(zip(labels.tolist(), TRUTH.tolist()))
    return len(pairs) == len(set(labels.tolist())) == len(set(TRUTH.tolist()))


def _all_contiguous(labels):
    return all(reg.is_connected(np.flatnonzero(labels == r), ADJ) for r in np.unique(labels) if r >= 0)


def test_redcap_recovers_the_blocks_with_every_linkage():
    for linkage in reg.REDCAP_LINKAGES:
        out = reg.redcap(X, G, 4, linkage=linkage)
        assert _matches_truth(out["labels"]), linkage
        assert _all_contiguous(out["labels"])


def test_redcap_is_deterministic_and_respects_min_size():
    a = reg.redcap(X, G, 6)["labels"]
    b = reg.redcap(X, G, 6)["labels"]
    assert np.array_equal(a, b)
    sized = reg.redcap(X, G, 6, min_size=10)["labels"]
    assert min(np.bincount(sized)) >= 10


def test_azp_variants_recover_the_blocks_and_stay_contiguous():
    for method in ("greedy", "sa", "tabu"):
        out = reg.azp(X, G, 4, restarts=4, seed=3, method=method)
        assert _all_contiguous(out["labels"]), method
        assert out["k"] == 4
        assert out["ssd"] <= out["start_ssd_median"]
        assert _matches_truth(out["labels"]), method


def test_max_p_meets_the_floor_and_reaches_the_bound_on_the_blocks():
    floor = np.ones(SIDE * SIDE)
    out = reg.max_p_regions(X, G, floor, 36, iterations=30, seed=2)
    assert out["p"] == 4                                    # 144 / 36 is the upper bound
    assert all(s >= 36 for s in out["floor_by_region"])
    assert _all_contiguous(out["labels"])
    assert _matches_truth(out["labels"])


def test_max_p_local_search_never_breaks_the_floor():
    floor = np.random.default_rng(5).integers(1, 5, SIDE * SIDE).astype(float)
    out = reg.max_p_regions(X, G, floor, 40.0, iterations=20, seed=4, local_search="sa")
    assert all(s >= 40 for s in out["floor_by_region"])
    assert _all_contiguous(out["labels"])
    assert out["ssd"] <= out["construction_ssd"] + 1e-9


def test_max_p_refuses_an_unreachable_floor():
    try:
        reg.max_p_regions(X, G, np.ones(SIDE * SIDE), 10_000)
    except ValueError:
        return
    raise AssertionError("A floor no component can reach must be refused")


def test_running_region_sums_match_direct_ssd():
    labels = reg.redcap(X, G, 5)["labels"]
    stats = reg._RegionStats(reg.standardize(X), labels)
    assert abs(stats.total() - reg.total_ssd(reg.standardize(X), labels)) < 1e-9
    i = int(np.flatnonzero(labels == 0)[0])
    dst = 1
    before = stats.total()
    d = stats.move_delta(i, 0, dst)
    stats.move(i, 0, dst)
    assert abs(stats.total() - (before + d)) < 1e-9


def test_silhouette_matches_a_hand_computed_case():
    x = np.array([[0.0], [1.0], [10.0], [11.0]])
    sil = rq.silhouette_samples(x, np.array([0, 0, 1, 1]))
    z = (x - x.mean()) / x.std()
    a = abs(z[0, 0] - z[1, 0])
    b = (abs(z[0, 0] - z[2, 0]) + abs(z[0, 0] - z[3, 0])) / 2
    assert abs(sil[0] - (b - a) / b) < 1e-12


def test_quality_of_the_true_partition_beats_a_scrambled_one():
    good = rq.partition_summary(X, TRUTH, ADJ)
    bad_labels = np.random.default_rng(9).permutation(TRUTH)
    bad = rq.partition_summary(X, bad_labels, ADJ)
    assert good["r2"] > 0.9 and bad["r2"] < 0.1
    assert good["all_contiguous"] and not bad["all_contiguous"]
    assert rq.silhouette_samples(X, TRUTH).mean() > rq.silhouette_samples(X, bad_labels).mean()
    assert rq.boundary_silhouette(X, TRUTH, ADJ).max() > 0
    path = rq.path_silhouette(X, TRUTH, ADJ)
    assert path.mean() > 0


def test_homogeneity_completeness_extremes():
    same = rq.homogeneity_completeness(TRUTH, TRUTH)
    assert abs(same["v_measure"] - 1.0) < 1e-12
    finer = rq.homogeneity_completeness(np.arange(len(TRUTH)), TRUTH)
    assert abs(finer["homogeneity"] - 1.0) < 1e-12 and finer["completeness"] < 0.5


def test_isoperimetric_quotient_of_simple_shapes():
    assert abs(rq.isoperimetric_quotient(np.pi, 2 * np.pi) - 1.0) < 1e-12
    assert abs(rq.isoperimetric_quotient(1.0, 4.0) - np.pi / 4) < 1e-12


def run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("REGIONALIZATION SMOKE TESTS OK")


if __name__ == "__main__":
    run_all()
