# -*- coding: utf-8 -*-
"""QGIS-independent tests for core/local_stats_engines.py (v3.11.0).

These assert properties a correct implementation must have - symmetry,
known closed forms, invariance, the behaviour on constructed patterns - not
snapshots of whatever the code returned once.  The numerical agreement with
PySAL's esda is checked separately in tests/esda_parity.py, which needs the
QGIS Python environment where esda is installed.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("local_stats_engines", ROOT / "core" / "local_stats_engines.py")
lse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lse)


def lattice(rows: int, cols: int, queen: bool = True):
    neighbors, weights, ids = {}, {}, []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            ids.append(i)
            nb = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) == (0, 0) or (not queen and dr and dc):
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        nb.append(rr * cols + cc)
            neighbors[i] = nb
            weights[i] = [1.0 / len(nb)] * len(nb)
    return lse.SpatialGraph.from_dicts(neighbors, weights, ids)


G = lattice(10, 10)
RNG = np.random.default_rng(7)
ROWS = np.repeat(np.arange(10), 10).astype(float)
TREND = ROWS + RNG.normal(scale=0.5, size=100)
NOISE = RNG.normal(size=100)


def test_graph_drops_unusable_neighbours_and_self_links():
    g = lse.SpatialGraph.from_dicts({1: [1, 2, 3, 2], 2: [1], 3: [9]}, {1: [1, 1, 1, 1], 2: [1], 3: [1]}, [1, 2, 3])
    assert g.cardinality.tolist() == [2, 1, 0]          # self, duplicate and unknown id removed
    assert g.islands.tolist() == [2]
    assert np.allclose(g.row_standardized().rowsum[:2], 1.0)


def test_conditional_draws_never_include_the_focal_unit_or_repeat():
    for i, draws, w in lse.conditional_draws(G, 50, 3):
        assert not np.any(draws == i)
        assert all(len(set(row)) == len(row) for row in draws)
        assert draws.shape == (50, len(w))


def test_two_sided_p_is_symmetric_and_bounded():
    sims = np.arange(-50, 51, dtype=float)
    assert lse.two_sided_p(1000.0, sims) == lse.two_sided_p(-1000.0, sims)
    assert lse.two_sided_p(1000.0, sims) == 2.0 / 102.0
    assert lse.two_sided_p(0.0, sims) == 1.0


def test_chi2_sf_matches_closed_forms():
    # dof = 2: sf(x) = exp(-x/2); dof = 1: sf(x) = erfc(sqrt(x/2))
    for x in (0.5, 2.0, 7.3):
        assert abs(lse.chi2_sf(x, 2.0) - math.exp(-x / 2)) < 1e-12
        assert abs(lse.chi2_sf(x, 1.0) - math.erfc(math.sqrt(x / 2))) < 1e-10
    assert abs(lse.chi2_sf(3.84145882, 1.0) - 0.05) < 1e-6


def test_global_eb_shrinks_small_bases_more_and_keeps_the_overall_rate():
    base = np.array([10, 20, 5000, 8000, 30, 6000, 15, 9000], dtype=float)
    events = np.array([3, 0, 60, 70, 2, 45, 1, 95], dtype=float)
    out = lse.global_eb_rate(events, base)
    assert out["between_variance"] > 0
    w = out["shrink_weight"]
    assert w[np.argmin(base)] < w[np.argmax(base)]
    raw = events / base
    # every EB rate lies between its raw rate and the overall rate
    lo, hi = np.minimum(raw, out["overall_rate"]), np.maximum(raw, out["overall_rate"])
    assert np.all(out["rate"] >= lo - 1e-15) and np.all(out["rate"] <= hi + 1e-15)


def test_global_eb_clamps_negative_between_variance():
    base = np.full(6, 1000.0)
    events = np.array([10, 10, 10, 10, 10, 11], dtype=float)   # far less spread than Poisson
    out = lse.global_eb_rate(events, base)
    assert out["variance_clamped"]
    assert np.allclose(out["rate"], out["overall_rate"])


def test_rates_reject_non_positive_base():
    try:
        lse.raw_rate([1, 2], [1, 0])
    except ValueError:
        return
    raise AssertionError("A zero base must be refused")


def test_local_moran_sums_to_global_and_finds_the_gradient():
    out = lse.moran_local_perm(TREND, G, 199, 1)
    z = TREND - TREND.mean()
    g_r = G.row_standardized()
    global_i = len(z) / g_r.weight.sum() * float(z @ g_r.lag(z)) / float(z @ z)
    assert abs(out["I"].sum() / (len(z) - 1) - global_i) < 1e-12
    hh = [q for q, r in zip(out["quadrant"], ROWS) if r == 9]
    assert hh.count("HH") >= 8


def test_lee_l_is_symmetric_and_differs_from_bivariate_moran():
    a = lse.lee_l(TREND, NOISE + TREND, G, 49, 1)
    b = lse.lee_l(NOISE + TREND, TREND, G, 49, 1)
    assert abs(a["L"] - b["L"]) < 1e-12
    assert np.allclose(a["local"], b["local"])
    zx = (TREND - TREND.mean()) / TREND.std()
    zy = (NOISE + TREND - (NOISE + TREND).mean()) / (NOISE + TREND).std()
    bivariate_moran = zx * G.row_standardized().lag(zy)
    assert not np.allclose(a["local"], bivariate_moran)


def test_lee_l_self_window_is_an_equal_share():
    # A unit with k neighbours must weigh itself 1/(k+1) - not 1/2, which is what
    # adding a self weight of 1 to already row-standardised weights would give.
    one_hot = np.zeros(100)
    one_hot[55] = 1.0
    out = lse.lee_l(one_hot, one_hot, G, 9, 1, include_self=True)
    z = (one_hot - one_hot.mean()) / one_hot.std()
    expected = z[55] / 9 + z[np.arange(100) != 55][0] * 8 / 9
    assert abs(out["smoothed_x"][55] - expected) < 1e-12


def test_multivariate_geary_matches_a_hand_computed_interior_value():
    # Two copies of the row index: z-scores are identical, so c_i is the plain
    # local Geary of ROWS/std. An interior unit has 8 neighbours, 6 of them one
    # row away: c = (6/8) / var(ROWS).
    x = np.column_stack([ROWS, ROWS * 2.0])
    out = lse.multivariate_local_geary(x, G, 19, 1)
    interior = [r * 10 + c for r in range(1, 9) for c in range(1, 9)]
    assert np.allclose(out["c"][interior], 0.75 / ROWS.var())


def test_local_join_counts_count_exactly_and_skip_ineligible_units():
    x = np.zeros(100)
    x[[0, 1, 10, 11]] = 1.0                       # a 2x2 block in the corner
    out = lse.local_join_counts(x, G, 199, 1)
    assert out["ljc"][0] == 3 and out["ljc"][11] == 3
    assert np.isnan(out["p"][50]) and out["ljc"][50] == 0
    assert out["p"][0] < 0.05


def test_losh_is_one_on_average_and_refuses_flat_fields():
    out = lse.losh(NOISE, G, 2.0, True)
    assert abs(float(np.mean(out["H"])) - 1.0) < 0.2
    out1 = lse.losh(NOISE, G, 1.0, True)
    assert np.all(np.isnan(out1["p"]))               # no inference outside a = 2
    try:
        lse.losh(np.ones(100), G)
    except ValueError:
        return
    raise AssertionError("A constant field must be refused")


def test_conditional_moran_removes_a_pattern_the_control_explains():
    y = TREND + 0.05 * NOISE
    plain = lse.moran_local_perm(y, G, 999, 1)
    cond = lse.moran_local_conditional(y, ROWS.reshape(-1, 1), G, 999, 1)
    assert cond["controls_r2"] > 0.9
    # 59 units are significant on the gradient; once the row index is
    # regressed out only chance-level hits (about 5 of 100) may remain.
    assert np.sum(plain["p"] < 0.05) > 40
    assert np.sum(cond["p"] < 0.05) <= 10


def test_gamma_cross_product_and_difference_agree_in_direction():
    c = lse.gamma_index(TREND, G.binary(), "c", True, 199, 1)
    s = lse.gamma_index(TREND, G.binary(), "s", True, 199, 1)
    assert c["gamma"] > c["sim_mean"] and s["gamma"] < s["sim_mean"]
    assert c["p"] < 0.05 and s["p"] < 0.05


def test_smaup_falls_with_k_and_with_rho():
    s_few = lse.smaup_statistic(400, 10, 0.3)["smaup"]
    s_many = lse.smaup_statistic(400, 300, 0.3)["smaup"]
    assert s_few > s_many
    assert lse.smaup_statistic(400, 50, 0.8)["smaup"] < lse.smaup_statistic(400, 50, -0.5)["smaup"]
    # rho exactly on a band edge uses the nearest table row, not a fall-through
    assert lse.smaup_statistic(400, 50, 0.8)["rho_row"] in (0.7, 0.9)
    assert lse.smaup_statistic(400, 50, 1.0)["rho_row"] == 0.9


def test_getis_ord_star_flags_the_high_end_of_a_gradient():
    out = lse.getis_ord_star(TREND, G)
    assert np.mean(out["z"][ROWS == 9]) > 1.96 and np.mean(out["z"][ROWS == 0]) < -1.96


def run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("LOCAL STATS SMOKE TESTS OK")


if __name__ == "__main__":
    run_all()
