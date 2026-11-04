#!/usr/bin/env python3
r"""Numerical truth of the classic engines against hand formulas and exact nulls.

Run with a QGIS Python runtime, offscreen, a fresh profile and PYTHONPATH set
to the folder that holds ``planx_geostats``, e.g.

    set QT_QPA_PLATFORM=offscreen
    set PYTHONPATH=C:\...\qgis_plugins
    C:\OSGeo4W\bin\python-qgis-ltr.bat tests\test_numeric_truth.py

Every reference is computed here independently: dense-matrix formulas, the
exact permutation distribution (all n! orderings for n = 7) and Student's t
from scipy when present (fixed table values otherwise). The QGIS part builds
queen, rook, KNN and distance-band weights on a grid of squares. Exits 1 on
any failure.
"""
from __future__ import annotations

import importlib
import itertools
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# The engines are loaded as a synthetic package over core/ so that their
# relative imports work without importing the plugin (and its GUI).
_PKG = types.ModuleType("gs_truth")
_PKG.__path__ = [str(ROOT / "core")]
sys.modules["gs_truth"] = _PKG
sys.dont_write_bytecode = True
se = importlib.import_module("gs_truth.stats_engines")
lse = importlib.import_module("gs_truth.local_stats_engines")
calculate_geary_c = importlib.import_module("gs_truth.advanced_stats_engines").calculate_geary_c

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  | {detail}" if detail and not ok else ""))


def close(a, b, tol=1e-10):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def lattice(side, rook=False):
    """Neighbour dicts of a side x side lattice (ids 0..n-1, row-major)."""
    nb = {}
    for r in range(side):
        for c in range(side):
            i = r * side + c
            out = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) == (0, 0) or (rook and dr and dc):
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < side and 0 <= cc < side:
                        out.append(rr * side + cc)
            nb[i] = out
    return nb


def row_std(nb):
    return {i: [1.0 / len(v)] * len(v) if v else [] for i, v in nb.items()}


def ones(nb):
    return {i: [1.0] * len(v) for i, v in nb.items()}


def dense(nb, wt, n):
    W = np.zeros((n, n))
    for i, js in nb.items():
        for j, w in zip(js, wt[i]):
            W[i, j] = w
    return W


SIDE = 4
N = SIDE * SIDE
IDS = list(range(N))
Q = lattice(SIDE)
QW = row_std(Q)
WQ = dense(Q, QW, N)
Y = np.array([1, 2, 3, 4, 2, 3, 5, 8, 3, 6, 9, 12, 4, 9, 13, 20], dtype=float)


def moran_ref(y, W):
    n = len(y)
    z = y - y.mean()
    S0 = W.sum()
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(1) + W.sum(0)) ** 2).sum()
    I = n / S0 * (z @ W @ z) / (z @ z)
    b2 = n * (z ** 4).sum() / (z @ z) ** 2
    EI = -1.0 / (n - 1)
    VI = (n * ((n * n - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
          - b2 * ((n * n - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)) / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2) - EI ** 2
    return I, EI, VI, (I - EI) / math.sqrt(VI)


def exact_permutation_moments(stat, y):
    vals = np.array([stat(np.array(p)) for p in itertools.permutations(y)])
    return vals.mean(), vals.var()


# ---------------------------------------------------------------------------
# Global statistics
# ---------------------------------------------------------------------------


def test_global_moran():
    I, EI, VI, zI = moran_ref(Y, WQ)
    mi, ei, vi, zi, pi = se.calculate_global_moran(Y, Q, QW, IDS)
    check("global Moran I = hand formula", close(mi, I, 1e-12), f"{mi} vs {I}")
    check("global Moran E[I] = -1/(n-1)", close(ei, EI, 1e-15))
    check("global Moran VI (randomisation) = hand formula", close(vi, VI, 1e-12), f"{vi} vs {VI}")
    check("global Moran z", close(zi, zI, 1e-10))
    check("global Moran p two-sided normal", close(pi, math.erfc(abs(zI) / math.sqrt(2)), 1e-12))
    # Exact null for n = 7 on a path with asymmetric weights.
    y7 = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.6])
    nb7 = {i: [j for j in (i - 1, i + 1, i + 3) if 0 <= j < 7] for i in range(7)}
    wt7 = {i: [1.0 / (1 + abs(i - j)) for j in nb7[i]] for i in range(7)}
    W7 = dense(nb7, wt7, 7)
    z7 = y7 - y7.mean()
    stat = lambda v: 7 / W7.sum() * ((v - v.mean()) @ W7 @ (v - v.mean())) / (z7 @ z7)  # noqa: E731
    mean7, var7 = exact_permutation_moments(stat, y7)
    _, ei7, vi7, _, _ = se.calculate_global_moran(y7, nb7, wt7, list(range(7)))
    check("global Moran E[I] = exact permutation mean (n=7)", close(ei7, mean7, 1e-12), f"{ei7} vs {mean7}")
    check("global Moran VI = exact permutation variance (n=7)", close(vi7, var7, 1e-12), f"{vi7} vs {var7}")


def general_g_dense(x, W):
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    return (x @ W @ x) / (x.sum() ** 2 - (x ** 2).sum()), W.sum() / (len(x) * (len(x) - 1))


def test_general_g_exact():
    x7 = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.6])
    nb7 = {i: [j for j in (i - 1, i + 1, i + 3) if 0 <= j < 7] for i in range(7)}
    for label, wt7 in (("binary", ones(nb7)), ("row-standardised", row_std(nb7))):
        W7 = dense(nb7, wt7, 7)
        den = x7.sum() ** 2 - (x7 ** 2).sum()
        mean7, var7 = exact_permutation_moments(lambda v: v @ W7 @ v / den, x7)
        g, eg, vg, zg, pg = se.calculate_general_g(x7, nb7, wt7, list(range(7)))
        G_ref, _ = general_g_dense(x7, W7)
        check(f"General G = x'Wx / (sum^2 - sum sq) ({label})", close(g, G_ref, 1e-12))
        check(f"General G E[G] = exact permutation mean ({label}, n=7)", close(eg, mean7, 1e-12))
        check(f"General G VG = exact permutation variance ({label}, n=7)", close(vg, var7, 1e-12),
              f"{vg} vs {var7}")
        check(f"General G z and p ({label})", close(zg, (g - eg) / math.sqrt(var7), 1e-8)
              and close(pg, math.erfc(abs(zg) / math.sqrt(2)), 1e-12))


def test_general_g_sparse():
    rng = np.random.default_rng(11)
    n = 300
    pts = rng.uniform(0, 1000, size=(n, 2))
    x = rng.gamma(2.0, size=n) + pts[:, 0] / 500.0
    band = 90.0
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    Wb = ((d <= band) & ~np.eye(n, dtype=bool)).astype(float)
    rows, cols = se.distance_band_pairs(pts, band)
    ref_r, ref_c = np.nonzero(Wb)
    check("distance_band_pairs = brute-force pair set",
          rows.size == ref_r.size and np.array_equal(np.lexsort((cols, rows)), np.arange(rows.size))
          and set(zip(rows.tolist(), cols.tolist())) == set(zip(ref_r.tolist(), ref_c.tolist())))
    g = se.calculate_general_g_distance_band(x, pts, band)
    nb = {i: list(np.flatnonzero(Wb[i])) for i in range(n)}
    g2 = se.calculate_general_g(x, nb, ones(nb), list(range(n)))
    G_ref, EG_ref = general_g_dense(x, Wb)
    check("General G distance-band helper = dict path", close(g, g2, 1e-12))
    check("General G sparse G, E[G] = dense", close(g[:2], (G_ref, EG_ref), 1e-12))
    # Asymmetric KNN weights exercise A2 (w_ij w_ji) and unequal row/col sums.
    knn = {i: list(np.argsort(d[i])[1:5]) for i in range(n)}
    kw = {i: [1.0 / (1.0 + d[i, j]) for j in knn[i]] for i in range(n)}
    Wk = dense(knn, kw, n)
    out = se.calculate_general_g(x, knn, kw, list(range(n)))
    S0 = Wk.sum(); A1 = (Wk ** 2).sum(); A2 = (Wk * Wk.T).sum()
    r, c = Wk.sum(1), Wk.sum(0)
    A3 = (r ** 2).sum() + (c ** 2).sum(); A4 = (r * c).sum()
    s1, s2, s3, s4 = x.sum(), (x ** 2).sum(), (x ** 3).sum(), (x ** 4).sum()
    den = s1 ** 2 - s2
    t1 = (A1 + A2) * (s2 ** 2 - s4) / (n * (n - 1))
    t2 = (A3 + 2 * A4 - 2 * A1 - 2 * A2) * (s2 * (s1 ** 2 - s2) - 2 * s1 * s3 + 2 * s4) / (n * (n - 1) * (n - 2))
    t3 = (S0 ** 2 - A3 - 2 * A4 + A1 + A2) * (s1 ** 4 - 6 * s1 ** 2 * s2 + 8 * s1 * s3 + 3 * s2 ** 2 - 6 * s4) \
        / (n * (n - 1) * (n - 2) * (n - 3))
    vg = (t1 + t2 + t3) / den ** 2 - (S0 / (n * (n - 1))) ** 2
    check("General G sparse VG = dense Getis-Ord 1992 (asymmetric KNN)", close(out[2], vg, 1e-14),
          f"{out[2]} vs {vg}")
    # 20 000 points: must not build an n x n matrix (that alone is 3.2 GB).
    big = rng.uniform(0, 20000, size=(20000, 2))
    xb = rng.gamma(2.0, size=20000)
    t0 = time.time()
    res = se.calculate_general_g_distance_band(xb, big, 150.0)
    secs = time.time() - t0
    check("General G with 20 000 points is sparse (finite, < 10 s)", np.all(np.isfinite(res)) and secs < 10,
          f"{secs:.1f} s")


def test_geary():
    num = sum(WQ[i, j] * (Y[i] - Y[j]) ** 2 for i in range(N) for j in range(N))
    z = Y - Y.mean()
    C = (N - 1) * num / (2 * WQ.sum() * (z @ z))
    P = 199
    gc = calculate_geary_c(Y, Q, QW, IDS, permutations=P, seed=5)
    check("Geary C = hand formula", close(gc["observed_c"], C, 1e-12))
    # Independent replay of the documented p: (#|C_perm - mean| >= |C - mean| + 1) / (P + 1).
    rng = np.random.default_rng(5)
    i_idx, j_idx = np.nonzero(WQ)
    sims = []
    for _ in range(P):
        v = rng.permutation(Y)
        sims.append((N - 1) * np.sum(WQ[i_idx, j_idx] * (v[i_idx] - v[j_idx]) ** 2)
                    / (2 * WQ.sum() * np.sum((v - v.mean()) ** 2)))
    sims = np.array(sims)
    extreme = int(np.sum(np.abs(sims - sims.mean()) >= abs(C - sims.mean())))
    check("Geary permutation p = (extreme + 1) / (P + 1)", close(gc["p_value"], (extreme + 1) / (P + 1), 1e-15),
          f"{gc['p_value']} vs {(extreme + 1) / (P + 1)}")
    check("Geary p never below 1 / (P + 1)", gc["p_value"] >= 1.0 / (P + 1))


# ---------------------------------------------------------------------------
# Local statistics
# ---------------------------------------------------------------------------


def local_moran_ref(y, W):
    n = len(y)
    z = y - y.mean()
    m2 = (z @ z) / n
    lag = W @ z
    Ii = z / m2 * lag
    wi, wi2 = W.sum(1), (W ** 2).sum(1)
    b2 = n * (z ** 4).sum() / (z @ z) ** 2
    EIi = -wi / (n - 1)
    VIi = wi2 * (n - b2) / (n - 1) + (wi ** 2 - wi2) * (2 * b2 - n) / ((n - 1) * (n - 2)) - EIi ** 2
    return Ii, (Ii - EIi) / np.sqrt(VIi), z, lag


def lisa_labels(z, lag, zi):
    out = []
    for a, b, v in zip(z, lag, zi):
        if math.erfc(abs(v) / math.sqrt(2)) >= 0.05:
            out.append("Not Significant")
        else:
            out.append(("H" if a > 0 else "L") + ("H" if b > 0 else "L"))
    return out


def test_local_moran():
    Ii, zIi, z, lag = local_moran_ref(Y, WQ)
    li, lz, lp, lq = se.calculate_local_moran(Y, Q, QW, IDS)
    check("local Moran I_i = Anselin 1995", close(li, Ii, 1e-12))
    check("local Moran z_i", close(lz, zIi, 1e-10))
    check("local Moran labels (4x4)", list(lq) == lisa_labels(z, lag, zIi))
    # Exact null: all 7! orderings of a 7-unit chain, total randomisation.
    y7 = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.6])
    nb7 = {i: sorted({j for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < 7}) for i in range(7)}
    W7 = dense(nb7, row_std(nb7), 7)
    perms = np.array(list(itertools.permutations(y7)))
    Zp = perms - y7.mean()
    Ip = Zp * (Zp @ W7.T) / ((Zp[0] ** 2).sum() / 7)
    li7, lz7, _, _ = se.calculate_local_moran(y7, nb7, row_std(nb7), list(range(7)))
    check("local Moran z_i = exact permutation moments (n=7)",
          close(lz7, (li7 - Ip.mean(0)) / np.sqrt(Ip.var(0)), 1e-9))
    # A field with significant HL and LH outliers: one high value in a low
    # block and one low value in a high block on a 9 x 9 rook lattice.
    side = 9
    nb = lattice(side, rook=True)
    wt = row_std(nb)
    v = np.zeros(side * side)
    v[: side * side // 2] = 1.0
    v[side * side // 2:] = 10.0
    v[10] = 30.0  # high among lows -> HL
    v[70] = -20.0  # low among highs -> LH
    W = dense(nb, wt, side * side)
    Ii, zIi, z, lag = local_moran_ref(v, W)
    _, _, _, labels = se.calculate_local_moran(v, nb, wt, list(range(side * side)))
    ref = lisa_labels(z, lag, zIi)
    check("local Moran HL outlier labelled HL", labels[10] == "HL" == ref[10], f"{labels[10]} vs {ref[10]}")
    check("local Moran LH outlier labelled LH", labels[70] == "LH" == ref[70], f"{labels[70]} vs {ref[70]}")
    check("local Moran labels (outlier field) = reference", list(labels) == ref)


def gistar_ref(y, W):
    n = len(y)
    Wb = (W > 0).astype(float) + np.eye(n)
    xbar = y.mean()
    S = math.sqrt((y ** 2).mean() - xbar ** 2)
    Wi, S1i = Wb.sum(1), (Wb ** 2).sum(1)
    return (Wb @ y - xbar * Wi) / (S * np.sqrt((n * S1i - Wi ** 2) / (n - 1)))


def gi_ref(y, W):
    """Ord & Getis (1995) Gi: x_i excluded from the sums, mean and SD."""
    n = len(y)
    Wb = (W > 0).astype(float)
    out = np.full(n, np.nan)
    for i in range(n):
        xo = np.delete(y, i)
        wo = np.delete(Wb[i], i)
        if wo.sum() == 0:
            continue
        mo = xo.mean()
        so = math.sqrt((xo ** 2).mean() - mo ** 2)
        out[i] = (wo @ xo - mo * wo.sum()) / (so * math.sqrt(((n - 1) * (wo ** 2).sum() - wo.sum() ** 2) / (n - 2)))
    return out


def test_getis_ord():
    ref = gistar_ref(Y, WQ)
    gz, gp, gb = se.calculate_getis_ord(Y, Q, QW, IDS, star=True)
    check("Gi* z = Ord & Getis 1995 (self included)", close(gz, ref, 1e-10))
    check("Gi* p two-sided normal", close(gp, [math.erfc(abs(v) / math.sqrt(2)) for v in ref], 1e-12))
    exp_bins = [0 if p >= 0.10 else (3 if p < 0.01 else 2 if p < 0.05 else 1) * (1 if z > 0 else -1)
                for z, p in zip(ref, gp)]
    check("Gi* confidence bins", list(gb) == exp_bins)
    check("Gi* (local_stats_engines) = reference",
          close(lse.getis_ord_star(Y, lse.SpatialGraph.from_dicts(Q, QW, IDS))["z"], ref, 1e-10))
    gz0, gp0, _ = se.calculate_getis_ord(Y, Q, QW, IDS, star=False)
    check("Gi (no star) z = Ord & Getis 1995 (x_i excluded)", close(gz0, gi_ref(Y, WQ), 1e-10))
    # An island: feature 16 far away with no neighbour.
    y2 = np.append(Y, 50.0)
    nb2 = dict(Q)
    nb2[16] = []
    wt2 = dict(QW)
    wt2[16] = []
    ids2 = IDS + [16]
    gz2, gp2, gb2 = se.calculate_getis_ord(y2, nb2, wt2, ids2, star=True)
    check("Gi* island: z and p are NaN, bin 0", math.isnan(gz2[16]) and math.isnan(gp2[16]) and gb2[16] == 0,
          f"z={gz2[16]} p={gp2[16]} bin={gb2[16]}")
    check("Gi* non-islands unaffected by the island's NaN", np.all(np.isfinite(gz2[:16])))
    W2 = dense(nb2, wt2, 17)
    ref2 = gistar_ref(y2, W2)
    check("Gi* with an island: others = reference", close(gz2[:16], ref2[:16], 1e-10))
    gl = lse.getis_ord_star(y2, lse.SpatialGraph.from_dicts(nb2, wt2, ids2))
    check("Gi* (local_stats_engines) island NaN", math.isnan(gl["z"][16]) and math.isnan(gl["p"][16]))
    gz3, gp3, _ = se.calculate_getis_ord(y2, nb2, wt2, ids2, star=False)
    check("Gi (no star) island NaN", math.isnan(gz3[16]) and math.isnan(gp3[16]))


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


T_TABLE = (  # (t, df, two-sided p) from scipy.stats.t.sf, 1.17
    (2.0, 5, 0.10193947882985835),
    (2.228138851986, 10, 0.05000000000002326),
    (0.5, 3, 0.651447964848151),
    (3.5, 1, 0.17717106556580953),
    (1.96, 200, 0.051384840845813295),
    (-2.6, 9, 0.028738227044155587),
)


def test_ols():
    for t, df, p in T_TABLE:
        check(f"Student t p (t={t}, df={df})", close(se.student_t_two_sided_p(t, df), p, 1e-12),
              f"{se.student_t_two_sided_p(t, df)} vs {p}")
    rng = np.random.default_rng(5)
    n = 12
    X = rng.normal(size=(n, 2))
    y = 1 + 0.6 * X[:, 0] + rng.normal(size=n)
    side_nb = {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)}
    res = se.calculate_ols(y, X, side_nb, row_std(side_nb), list(range(n)), ["a", "b"])
    Xd = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
    e = y - Xd @ beta
    s2 = e @ e / (n - 3)
    se_b = np.sqrt(np.diag(s2 * np.linalg.inv(Xd.T @ Xd)))
    t = beta / se_b
    try:
        from scipy import stats
        p_ref = 2 * stats.t.sf(np.abs(t), n - 3)
    except ImportError:
        p_ref = np.array([se.student_t_two_sided_p(v, n - 3) for v in t])
    check("OLS coefficients = least squares", close(res["coefficients"], beta, 1e-10))
    check("OLS p-values = Student t with df_err (n=12)", close(res["p_values"], p_ref, 1e-10),
          f"{res['p_values']} vs {p_ref}")
    W = dense(side_nb, row_std(side_nb), n)
    I_ref = n / W.sum() * (e @ W @ e) / (e @ e)
    check("OLS residual Moran = (n/S0) e'We / e'e", close(res["residuals_moran"], I_ref, 1e-10))
    scaled = {i: [3.0 * w for w in v] for i, v in row_std(side_nb).items()}
    res_b = se.calculate_ols(y, X, side_nb, scaled, list(range(n)), ["a", "b"])
    check("OLS residual Moran invariant to weight scale (n/S0)", close(res_b["residuals_moran"], I_ref, 1e-10))


# ---------------------------------------------------------------------------
# C1: InputError
# ---------------------------------------------------------------------------


def test_input_errors():
    check("InputError is a ValueError", issubclass(se.InputError, ValueError))
    no_nb = {i: [] for i in IDS}
    no_w = {i: [] for i in IDS}
    const = np.full(N, 5.0)
    cases = [
        ("global Moran", lambda yy, nb, wt: se.calculate_global_moran(yy, nb, wt, IDS)),
        ("local Moran", lambda yy, nb, wt: se.calculate_local_moran(yy, nb, wt, IDS)),
        ("Gi*", lambda yy, nb, wt: se.calculate_getis_ord(yy, nb, wt, IDS)),
        ("Gi", lambda yy, nb, wt: se.calculate_getis_ord(yy, nb, wt, IDS, star=False)),
        ("General G", lambda yy, nb, wt: se.calculate_general_g(yy, nb, ones(nb), IDS)),
        ("Geary C", lambda yy, nb, wt: calculate_geary_c(yy, nb, wt, IDS, permutations=9)),
        ("Gi* (graph)", lambda yy, nb, wt: lse.getis_ord_star(yy, lse.SpatialGraph.from_dicts(nb, wt, IDS))),
    ]
    zero_w = {i: [0.0] * len(v) for i, v in Q.items()}
    for name, fn in cases:
        for what, args in (("all islands", (Y, no_nb, no_w)), ("constant field", (const, Q, QW)),
                           ("NaN value", (np.where(np.arange(N) == 3, np.nan, Y), Q, QW))):
            try:
                fn(*args)
                check(f"{name}: {what} raises InputError", False, "no exception")
            except ValueError as exc:
                text = str(exc)
                ok = (type(exc).__name__ == "InputError" and text.endswith(".") and text[:1].isupper()
                      and "nan" not in text.lower() and len(text.split()) >= 8)
                check(f"{name}: {what} raises InputError with a sentence", ok, repr(exc))
        if name in ("global Moran", "local Moran", "Geary C", "General G"):
            try:
                if name == "General G":
                    se.calculate_general_g(Y, Q, zero_w, IDS)
                else:
                    fn(Y, Q, zero_w)
                check(f"{name}: S0 = 0 raises InputError", False, "no exception")
            except ValueError as exc:
                check(f"{name}: S0 = 0 raises InputError", type(exc).__name__ == "InputError", repr(exc))
    # Every pair is a neighbour: General G has no variance under permutation.
    full = {i: [j for j in IDS if j != i] for i in IDS}
    try:
        se.calculate_general_g(Y, full, ones(full), IDS)
        check("General G complete graph raises InputError", False, "no exception")
    except se.InputError:
        check("General G complete graph raises InputError", True)


# ---------------------------------------------------------------------------
# Weights (QGIS)
# ---------------------------------------------------------------------------


def test_weights_qgis():
    from qgis.core import QgsApplication, QgsFeature, QgsGeometry, QgsPointXY, QgsVectorLayer

    app = QgsApplication([], False)
    app.initQgis()
    build_weights_matrix = importlib.import_module("gs_truth.weights").build_weights_matrix

    def grid(nx, ny, size=100.0):
        lyr = QgsVectorLayer("Polygon?crs=EPSG:3857&field=v:double", "g", "memory")
        feats = []
        for r in range(ny):
            for c in range(nx):
                x0, y0 = c * size, r * size
                f = QgsFeature(lyr.fields())
                f.setGeometry(QgsGeometry.fromPolygonXY([[
                    QgsPointXY(x0, y0), QgsPointXY(x0 + size, y0), QgsPointXY(x0 + size, y0 + size),
                    QgsPointXY(x0, y0 + size), QgsPointXY(x0, y0)]]))
                feats.append(f)
        lyr.dataProvider().addFeatures(feats)
        lyr.updateExtents()
        return lyr

    lyr = grid(4, 4)
    q, qw, ids, _ = build_weights_matrix(lyr, "queen")
    r, rw, _, _ = build_weights_matrix(lyr, "rook")
    pos = {fid: k for k, fid in enumerate(ids)}
    q_ref, r_ref = lattice(4), lattice(4, rook=True)
    check("queen neighbours = lattice (4x4)",
          all(sorted(pos[j] for j in q[f]) == sorted(q_ref[pos[f]]) for f in ids))
    check("rook neighbours = lattice, corners excluded (4x4)",
          all(sorted(pos[j] for j in r[f]) == sorted(r_ref[pos[f]]) for f in ids))
    check("rook cardinalities 4x2, 8x3, 4x4", sorted(len(v) for v in r.values()) == [2] * 4 + [3] * 8 + [4] * 4)
    check("queen cardinalities 4x3, 8x5, 4x8", sorted(len(v) for v in q.values()) == [3] * 4 + [5] * 8 + [8] * 4)
    check("queen weights row-standardised (each 1/k, rows sum to 1)",
          all(np.allclose(qw[f], 1.0 / len(q[f])) and abs(sum(qw[f]) - 1) < 1e-12 for f in ids))
    check("rook weights row-standardised", all(abs(sum(rw[f]) - 1) < 1e-12 for f in ids))
    k, kw, _, _ = build_weights_matrix(lyr, "knn", k_neighbors=3)
    check("KNN k=3: three neighbours, never itself", all(len(v) == 3 and f not in v for f, v in k.items()))
    d, dw, _, _ = build_weights_matrix(lyr, "distance", distance_band=100.0)
    check("distance band 100 m on 100 m cells = rook",
          all(sorted(d[f]) == sorted(r[f]) for f in ids))
    # Engines on QGIS-built weights reproduce the lattice reference.
    mi = se.calculate_global_moran(Y, q, qw, ids)[0]
    check("global Moran on QGIS queen weights = lattice reference", close(mi, moran_ref(Y, WQ)[0], 1e-12))
    app.exitQgis()


def main() -> int:
    for fn in (test_global_moran, test_general_g_exact, test_general_g_sparse, test_geary, test_local_moran,
               test_getis_ord, test_ols, test_input_errors):
        try:
            fn()
        except Exception as exc:  # a crash is a failure of that group, not of the run
            check(f"{fn.__name__} ran to the end", False, repr(exc))
    try:
        import qgis.core  # noqa: F401
        has_qgis = True
    except ImportError:
        has_qgis = False
    if has_qgis:
        test_weights_qgis()
    else:
        print("SKIP weights: QGIS is not importable")
    failed = [n for n, ok in RESULTS if not ok]
    if failed:
        print(f"NUMERIC TRUTH: FAIL ({len(failed)} of {len(RESULTS)}): {', '.join(failed)}")
        return 1
    print(f"NUMERIC TRUTH: PASS ({len(RESULTS)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
