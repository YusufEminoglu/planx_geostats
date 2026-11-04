#!/usr/bin/env python3
r"""Numerical parity of core/local_stats_engines.py with PySAL esda.

Run with a QGIS Python runtime that has esda installed (both OSGeo4W runtimes
do), e.g.  C:\OSGeo4W\bin\python-qgis-ltr.bat tests\esda_parity.py

Every statistic must equal esda's to 1e-9 on the same data and the same
weights.  Permutation p-values cannot be equal - the two use different random
streams - so for those the check is that both runs order the units the same
way (rank correlation of the p-values above 0.9).  Without esda the suite
prints a SKIP line and exits 0: esda is an optional dependency of the plugin
and the engines do not use it.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import sys
import types
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-9


def _load_engine():
    spec = importlib.util.spec_from_file_location("local_stats_engines", ROOT / "core" / "local_stats_engines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_core_package():
    """stats_engines, advanced_stats_engines and weights as one synthetic package.

    The relative imports between the engines then work without importing the
    plugin itself (and its GUI).
    """
    package = types.ModuleType("gs_parity")
    package.__path__ = [str(ROOT / "core")]
    sys.modules["gs_parity"] = package
    return {name: importlib.import_module(f"gs_parity.{name}")
            for name in ("stats_engines", "advanced_stats_engines")}


def _grid_weights(side, kind):
    """Queen/rook weights built by core/weights.py on a QGIS grid of squares.

    Returns (neighbours, weights, ids in lattice order) or None without QGIS.
    """
    try:
        from qgis.core import QgsApplication, QgsFeature, QgsGeometry, QgsPointXY, QgsVectorLayer
    except ImportError:
        return None
    if QgsApplication.instance() is None:
        app = QgsApplication([], False)
        app.initQgis()
        _QGIS_APP.append(app)
    weights = importlib.import_module("gs_parity.weights")
    lyr = QgsVectorLayer("Polygon?crs=EPSG:3857", "g", "memory")
    feats = []
    for r in range(side):
        for c in range(side):
            x0, y0 = c * 10.0, r * 10.0
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPolygonXY([[
                QgsPointXY(x0, y0), QgsPointXY(x0 + 10, y0), QgsPointXY(x0 + 10, y0 + 10),
                QgsPointXY(x0, y0 + 10), QgsPointXY(x0, y0)]]))
            feats.append(f)
    lyr.dataProvider().addFeatures(feats)
    nb, wt, ids, _ = weights.build_weights_matrix(lyr, kind)
    pos = {fid: k for k, fid in enumerate(ids)}
    return ({pos[f]: [pos[j] for j in nb[f]] for f in ids},
            {pos[f]: list(wt[f]) for f in ids}, list(range(len(ids))))


_QGIS_APP = []


def _rank_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[keep]))
    rb = np.argsort(np.argsort(b[keep]))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    warnings.filterwarnings("ignore")
    try:
        import esda
        import libpysal
        from esda.moran_local_mv import MoranLocalConditional, MoranLocalPartial
        from esda.smoothing import Empirical_Bayes, Spatial_Empirical_Bayes, Spatial_Rate, assuncao_rate
    except ImportError as exc:
        print(f"ESDA PARITY: OK (skipped - esda not installed: {exc})")
        return 0

    lse = _load_engine()
    rng = np.random.default_rng(3)
    side = 12
    n = side * side

    def w_new(transform="o"):
        w = libpysal.weights.lat2W(side, side, rook=False)
        w.transform = transform
        return w

    w0 = w_new()
    ids = w0.id_order
    graph = lse.SpatialGraph.from_dicts({i: list(w0.neighbors[i]) for i in ids},
                                        {i: [1.0] * len(w0.neighbors[i]) for i in ids}, ids)
    row = np.array([i // side for i in range(n)], float)
    col = np.array([i % side for i in range(n)], float)
    y = 0.3 * row + rng.normal(size=n)
    x2 = 0.2 * col + 0.4 * y + rng.normal(size=n)
    x3 = rng.normal(size=n)
    base = rng.integers(50, 5000, n).astype(float)
    events = rng.binomial(base.astype(int), 0.01 + 0.004 * row / side).astype(float)

    failures = []

    def same(label, ours, theirs, tol=TOL):
        diff = float(np.nanmax(np.abs(np.asarray(ours, float).ravel() - np.asarray(theirs, float).ravel())))
        status = "ok" if diff <= tol else "FAIL"
        print(f"  {label:<38} max |diff| = {diff:.2e}  {status}")
        if diff > tol:
            failures.append(label)

    def ordered(label, ours, theirs, floor=0.9):
        rho = _rank_corr(ours, theirs)
        status = "ok" if rho >= floor else "FAIL"
        print(f"  {label:<38} rank corr  = {rho:.3f}     {status}")
        if rho < floor:
            failures.append(label)

    same("Assuncao-Reis rate", lse.assuncao_reis_rate(events, base), assuncao_rate(events, base))
    geb = lse.global_eb_rate(events, base)
    assert geb["between_variance"] > 0, "fixture must not trigger the clamp esda lacks"
    same("Global EB rate", geb["rate"], Empirical_Bayes(events, base).r)
    same("Spatial window rate", lse.spatial_rate(events, base, graph), Spatial_Rate(events, base, w_new()).r)
    same("Spatial EB rate", lse.spatial_eb_rate(events, base, graph)["rate"],
         Spatial_Empirical_Bayes(events, base, w_new()).r)

    ml = esda.Moran_Local(y, w_new("r"), permutations=999, seed=1)
    ours = lse.moran_local_perm(y, graph, 999, 1)
    same("Local Moran I", ours["I"], ml.Is)
    ordered("Local Moran p", ours["p"], ml.p_sim)
    same("Global Moran I", lse.moran_global_perm(y, graph, 9, 1)["I"], esda.Moran(y, w_new("r"), permutations=0).I)

    gmv = esda.Geary_Local_MV(connectivity=w_new(), permutations=999).fit([y, x2, x3])
    ours = lse.multivariate_local_geary(np.column_stack([y, x2, x3]), graph, 999, 1)
    same("Multivariate local Geary", ours["c"], gmv.localG)
    ordered("Multivariate local Geary p", ours["p"], gmv.p_sim)

    xb = (y > np.median(y)).astype(float)
    zb = (x2 > np.median(x2)).astype(float)
    x3b = (x3 > 0).astype(float)
    same("Local join count (univariate)", lse.local_join_counts(xb, graph, 9, 1)["ljc"],
         esda.Join_Counts_Local(connectivity=w_new()).fit(xb, permutations=0).LJC)
    for case in ("CLC", "BJC"):
        same(f"Local join count ({case})", lse.local_join_counts(xb, graph, 9, 1, z=zb, mode=case.lower())["ljc"],
             esda.Join_Counts_Local_BV(connectivity=w_new()).fit(xb, zb, case=case, permutations=0).LJC)
    same("Local join count (multivariate)",
         lse.local_join_counts(xb, graph, 9, 1, mode="multivariate", extra=[zb, x3b])["ljc"],
         esda.Join_Counts_Local_MV(connectivity=w_new()).fit([xb, zb, x3b], permutations=0).LJC)

    lo = esda.LOSH(connectivity=w_new(), inference="chi-square").fit(y)
    ours = lse.losh(y, graph, 2.0, include_self=False)
    same("LOSH H", ours["H"], lo.Hi)
    same("LOSH variance", ours["var"], lo.VarHi)
    same("LOSH chi-square p", ours["p"], lo.pval, tol=1e-8)

    wr = w_new("r")
    sp = esda.Spatial_Pearson(connectivity=wr.sparse, permutations=0).fit(y.reshape(-1, 1), x2.reshape(-1, 1))
    ours = lse.lee_l(y, x2, graph, 9, 1, include_self=False)
    same("Lee's L (spatial Pearson)", ours["L"], sp.association_[0, 1])
    same("Lee's L_XX", ours["L_xx"], sp.association_[0, 0])
    spl = esda.Spatial_Pearson_Local(connectivity=wr.sparse.toarray(), permutations=0).fit(
        y.reshape(-1, 1), x2.reshape(-1, 1))
    same("Local Lee's L", ours["local"], spl.associations_)

    X = np.column_stack([x2, x3])
    mp = MoranLocalPartial(permutations=0).fit(X, y, w_new("r"))
    ours = lse.moran_local_partial(y, X, graph, 9, 1)
    same("Partial local Moran", ours["association"], mp.association_)
    labels = np.array([{"HH": 1, "LH": 2, "LL": 3, "HL": 4}[s] for s in ours["label"]])
    same("Partial local Moran labels", labels, mp.labels_, tol=0)
    mc = MoranLocalConditional(permutations=0).fit(X, y.reshape(-1, 1), w_new("r"))
    ours = lse.moran_local_conditional(y, X, graph, 9, 1)
    same("Conditional local Moran", ours["association"], mc.association_)

    for op in "csa":
        same(f"Gamma ({op})", lse.gamma_index(y, graph, op, False, 9, 1)["gamma"],
             esda.Gamma(y, w_new(), operation=op, permutations=0).g, tol=1e-7)

    for n_, k_, rho in ((100, 10, 0.5), (400, 50, 0.32), (900, 300, -0.2), (60, 6, 0.55)):
        s = esda.Smaup(n_, k_, rho)
        o = lse.smaup_statistic(n_, k_, rho)
        same(f"S-maup n={n_} k={k_} rho={rho}", [o["smaup"], o["critical"][0.05]], [s.smaup, s.critical_05])

    # v3.12.0 partition-quality measures against esda and scikit-learn
    spec = importlib.util.spec_from_file_location("region_quality", ROOT / "core" / "region_quality.py")
    rq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rq)
    try:
        from sklearn import metrics
    except ImportError:
        metrics = None
    wq = libpysal.weights.lat2W(side, side, rook=True)
    adj = [sorted(wq.neighbors[i]) for i in range(n)]
    labels = ((col >= side // 2).astype(int) + 2 * (row >= side // 2))
    labels[[50, 51]] = labels[[50, 51]] ^ 1
    xq = np.column_stack([labels * 2.0 + rng.normal(size=n), rng.normal(size=n)])
    zq = rq._z(xq)
    if metrics is not None:
        same("Silhouette (sklearn)", rq.silhouette_samples(xq, labels), metrics.silhouette_samples(zq, labels))
        ref = (row >= side // 2).astype(int)
        hc = rq.homogeneity_completeness(labels, ref)
        same("Homogeneity / completeness / V (sklearn)",
             [hc["homogeneity"], hc["completeness"], hc["v_measure"]],
             [metrics.homogeneity_score(ref, labels), metrics.completeness_score(ref, labels),
              metrics.v_measure_score(ref, labels)])
    same("Boundary silhouette", rq.boundary_silhouette(xq, labels, adj), esda.boundary_silhouette(zq, labels, wq))
    same("Path silhouette", rq.path_silhouette(xq, labels, adj), esda.path_silhouette(zq, labels, wq))

    # v3.13.0 classic engines (core/stats_engines.py, advanced_stats_engines.py)
    # against esda on the same weights: global Moran, General G, Geary C, Gi*,
    # Gi, Local Moran, OLS p-values, and the QGIS-built contiguity weights.
    core = _load_core_package()
    se, adv = core["stats_engines"], core["advanced_stats_engines"]
    yc = 5.0 + y - y.min()  # General G needs non-negative values
    for rook in (False, True):
        tag = "rook" if rook else "queen"
        wl = libpysal.weights.lat2W(side, side, rook=rook)
        nbd = {i: list(wl.neighbors[i]) for i in ids}
        row_w = {i: [1.0 / len(v)] * len(v) for i, v in nbd.items()}
        bin_w = {i: [1.0] * len(v) for i, v in nbd.items()}
        built = _grid_weights(side, tag)
        if built is not None:
            same(f"{tag} neighbour sets (QGIS = libpysal)",
                 [len(set(built[0][i]) ^ set(nbd[i])) for i in ids], [0] * n, tol=0)
            same(f"{tag} neighbour counts (QGIS = libpysal)",
                 [len(built[0][i]) for i in ids], [wl.cardinalities[i] for i in ids], tol=0)
            same(f"{tag} row-standardised weights", [sum(built[1][i]) for i in ids], [1.0] * n, tol=1e-12)
            same(f"{tag} each weight 1/k", np.concatenate([built[1][i] for i in ids]),
                 np.concatenate([[1.0 / len(nbd[i])] * len(nbd[i]) for i in ids]), tol=1e-12)
        wl.transform = "r"
        m = esda.Moran(y, wl, permutations=0)
        mi, ei, vi, zi, pi = se.calculate_global_moran(y, nbd, row_w, ids)
        same(f"Global Moran I, E[I], VI, z ({tag}, r)", [mi, ei, vi, zi, pi],
             [m.I, m.EI, m.VI_rand, m.z_rand, m.p_rand])
        wl.transform = "b"
        g = esda.G(yc, wl, permutations=0)
        og, eg, vg, zg, pg = se.calculate_general_g(yc, nbd, bin_w, ids)
        same(f"General G, E[G] ({tag}, b)", [og, eg], [g.G, g.EG])
        same(f"General G VG relative ({tag}, b)", [vg / g.VG], [1.0], tol=1e-9)
        # esda's p_norm is one-sided; the plugin reports the two-sided p.
        same(f"General G z, two-sided p ({tag}, b)", [zg, pg], [g.z_norm, 2 * g.p_norm], tol=1e-8)
        wl.transform = "r"
        gc = esda.Geary(y, wl, permutations=0)
        same(f"Geary C ({tag}, r)", adv.calculate_geary_c(y, nbd, row_w, ids, permutations=9)["observed_c"], gc.C)
        wl.transform = "b"
        gs = esda.G_Local(y, wl, star=True, transform="B", permutations=0)
        gz, gp, _ = se.calculate_getis_ord(y, nbd, row_w, ids, star=True)
        same(f"Gi* z ({tag})", gz, gs.Zs)
        same(f"Gi* p two-sided ({tag})", gp, 2 * gs.p_norm, tol=1e-9)
        same(f"Gi* z (local_stats_engines, {tag})", lse.getis_ord_star(y, lse.SpatialGraph.from_dicts(
            nbd, row_w, ids))["z"], gs.Zs)
        g0 = esda.G_Local(y, wl, star=False, transform="B", permutations=0)
        same(f"Gi (no star) z ({tag})", se.calculate_getis_ord(y, nbd, row_w, ids, star=False)[0], g0.Zs)
        wl.transform = "r"
        ml = esda.Moran_Local(y, wl, permutations=0)
        li, lz, lp, lq = se.calculate_local_moran(y, nbd, row_w, ids)
        same(f"Local Moran I_i (x (n-1)/n = esda Is, {tag})", li * (n - 1) / n, ml.Is)
        # z_i is not compared: esda's analytical VI is not the exact
        # randomisation variance (tests/test_numeric_truth.py checks the
        # plugin's z_i against all 7! permutations instead).
        quad = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
        exp_q = [quad[int(c)] if p_ < 0.05 else "Not Significant"
                 for c, p_ in zip(ml.q, [math.erfc(abs(v) / math.sqrt(2)) for v in lz])]
        same(f"Local Moran quadrants vs esda q ({tag})", [int(a == b) for a, b in zip(lq, exp_q)], [1] * n, tol=0)
    # Asymmetric KNN weights: S1 must count pairs where only w_ji is non-zero.
    pts = np.column_stack([col, row]) + rng.uniform(-0.3, 0.3, size=(n, 2))
    wk = libpysal.weights.KNN(pts, k=3)
    wk.transform = "r"
    knb = {i: list(wk.neighbors[i]) for i in ids}
    kw = {i: list(wk.weights[i]) for i in ids}
    m = esda.Moran(y, wk, permutations=0)
    same("Global Moran VI (asymmetric KNN, r)", se.calculate_global_moran(y, knb, kw, ids)[2], m.VI_rand)
    wk.transform = "b"
    g = esda.G(yc, wk, permutations=0)
    vg = se.calculate_general_g(yc, knb, {i: [1.0] * 3 for i in ids}, ids)[2]
    same("General G VG relative (asymmetric KNN, b)", [vg / g.VG], [1.0], tol=1e-9)
    # OLS p-values: Student t with n - k - 1 degrees of freedom.
    try:
        from scipy import stats as sstats
    except ImportError:
        sstats = None
    if sstats is not None:
        for n_ols in (8, 12, 30):
            Xo = rng.normal(size=(n_ols, 2))
            yo = 1 + 0.6 * Xo[:, 0] + rng.normal(size=n_ols)
            res = se.calculate_ols(yo, Xo, {}, {}, list(range(n_ols)), ["a", "b"])
            same(f"OLS p-values = exact t (n={n_ols})", res["p_values"],
                 2 * sstats.t.sf(np.abs(res["t_statistics"]), res["df_err"]), tol=1e-10)

    if failures:
        print(f"ESDA PARITY: FAIL ({len(failures)} failed): {', '.join(failures)}")
        return 1
    print("ESDA PARITY: PASS")
    return 0


if __name__ == "__main__":
    _rc = main()
    for _app in _QGIS_APP:
        _app.exitQgis()
    sys.exit(_rc)
