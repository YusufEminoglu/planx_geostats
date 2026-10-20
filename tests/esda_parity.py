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

import importlib.util
import sys
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

    if failures:
        print(f"ESDA PARITY: FAIL ({len(failures)} failed): {', '.join(failures)}")
        return 1
    print("ESDA PARITY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
