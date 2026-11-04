#!/usr/bin/env python3
r"""Robustness of the hot spot / LISA family on awkward but ordinary inputs.

Each check reproduces a 3.12.1 audit finding (P2, P3, P5, P6, P8, P10-P12,
P14, P15) through Processing, the way a user meets it, and asserts the 3.13
behaviour: nothing is hidden, nothing reads as a result when no test was run,
and impossible weights are refused with a hint.

Run with a QGIS Python, a fresh profile and no display:
  set QGIS_CUSTOM_CONFIG_PATH=%TEMP%\gs_rob_profile
  set QT_QPA_PLATFORM=offscreen
  C:\OSGeo4W\bin\python-qgis-ltr.bat planx_geostats\tests\test_robustness_qgis.py --root C:\Users\YE\PyCharmMiscProject\qgis_plugins

``--root`` is the folder that contains the ``planx_geostats`` package, so the
same file can be pointed at an export of an older revision (red before green).
Prints one PASS/FAIL line per check and exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

PROVIDER = "planx_geostats"
CHECKS = []


def check(func):
    CHECKS.append(func)
    return func


# ---------------------------------------------------------------- fixtures --
def _square(x0, y0, size=100.0):
    from qgis.core import QgsPointXY

    return [QgsPointXY(x0, y0), QgsPointXY(x0 + size, y0), QgsPointXY(x0 + size, y0 + size),
            QgsPointXY(x0, y0 + size), QgsPointXY(x0, y0)]


def _layer(name, geometries, values, crs="EPSG:3857", geometry_type="Polygon", extra=None):
    """Memory layer with an integer ``uid`` (feature order) and a double ``v``."""
    from qgis.core import QgsFeature, QgsProject, QgsVectorLayer

    fields = "&field=uid:integer&field=v:double" + "".join(f"&field={n}:double" for n in (extra or {}))
    layer = QgsVectorLayer(f"{geometry_type}?crs={crs}{fields}", name, "memory")
    features = []
    for i, (geometry, value) in enumerate(zip(geometries, values)):
        feature = QgsFeature(layer.fields())
        feature.setGeometry(geometry)
        feature.setAttribute("uid", i)
        feature.setAttribute("v", value)
        for field_name, column in (extra or {}).items():
            feature.setAttribute(field_name, column[i])
        features.append(feature)
    layer.dataProvider().addFeatures(features)
    layer.updateExtents()
    QgsProject.instance().addMapLayer(layer)
    return layer


def _grid(n=5, size=100.0, x0=0.0, y0=0.0):
    from qgis.core import QgsGeometry

    return [QgsGeometry.fromPolygonXY([_square(x0 + c * size, y0 + r * size, size)])
            for r in range(n) for c in range(n)]


BASE = [float((i * 7) % 11 + (i // 5)) for i in range(25)]


def _island_layer():
    from qgis.core import QgsGeometry

    geometries = _grid() + [QgsGeometry.fromPolygonXY([_square(50000.0, 50000.0)])]
    return _layer("island", geometries, BASE + [99.0])


def _points_layer(crs="EPSG:3857", step=100.0, x0=0.0, y0=0.0):
    from qgis.core import QgsGeometry, QgsPointXY

    geometries = [QgsGeometry.fromPointXY(QgsPointXY(x0 + (i % 5) * step, y0 + (i // 5) * step)) for i in range(25)]
    return _layer("points", geometries, BASE, crs=crs, geometry_type="Point")


# ----------------------------------------------------------------- running --
class _Feedback:
    """QgsProcessingFeedback that records warnings and can cancel on demand."""

    def __new__(cls, cancel_at_progress=None):
        from qgis.core import QgsProcessingFeedback

        class Feedback(QgsProcessingFeedback):
            def __init__(self):
                super().__init__()
                self.warnings = []
                self.cancel_at = cancel_at_progress

            def pushWarning(self, message):
                self.warnings.append(message)
                super().pushWarning(message)

            def setProgress(self, progress):
                super().setProgress(progress)
                if self.cancel_at is not None and progress >= self.cancel_at:
                    self.cancel()

        return Feedback()


def _run(alg, params, cancel_at_progress=None):
    """``(result, output_layer, context, feedback)`` - output layer or None."""
    import processing
    from qgis.core import QgsProcessingContext, QgsProject

    context = QgsProcessingContext()
    context.setProject(QgsProject.instance())
    feedback = _Feedback(cancel_at_progress)
    result = processing.run(f"{PROVIDER}:{alg}", params, context=context, feedback=feedback)
    out = result.get("OUTPUT")
    layer = context.getMapLayer(out) if isinstance(out, str) else out
    return result, layer, context, feedback


def _refused(alg, params, *needles):
    """Assert the run raises a Processing error whose text has every needle."""
    from qgis.core import QgsProcessingException

    try:
        _run(alg, params)
    except QgsProcessingException as exc:
        text = str(exc)
        missing = [n for n in needles if n.lower() not in text.lower()]
        assert not missing, f"{alg}: refused, but the message lacks {missing}: {text[:300]}"
        assert "Traceback" not in text, f"{alg}: refusal carries a traceback: {text[:300]}"
        return text
    raise AssertionError(f"{alg}: expected a plain refusal, but the run completed")


def _report(tmp, name):
    return str(Path(tmp) / f"{name}.html")


def _uid(feature):
    """Fixture position from the geometry (5 x 5 grid of 100 m cells; the island is 25).

    Keyed on geometry, not the ``uid`` attribute, so a check still reaches its
    own assertion on a revision that loses input attributes.
    """
    point = feature.geometry().centroid().asPoint()
    if point.x() > 10000:
        return 25
    return int(point.y() // 100) * 5 + int(point.x() // 100)


def _features(layer):
    return sorted(layer.getFeatures(), key=_uid)


def _is_null(value):
    from qgis.core import NULL

    return value is None or value == NULL or (isinstance(value, float) and math.isnan(value))


def _common(layer, tmp, name, **extra):
    params = {"INPUT": layer, "FIELD": "v", "WEIGHT_TYPE": 0, "KNN": 4, "DISTANCE_BAND": 150.0,
              "OUTPUT": "TEMPORARY_OUTPUT", "HTML_REPORT": _report(tmp, name)}
    params.update(extra)
    return params


# ------------------------------------------------------------------ checks --
@check
def island_gi_star_is_no_neighbours(tmp):
    """P6: a far-away polygon must not become a 99 % hot spot from its own value."""
    layer = _island_layer()
    for mode in (0, 1, 2):
        _res, out, _ctx, _fb = _run("getis_ord_gi", _common(layer, tmp, f"gi_island_{mode}", MULTIPLE_TESTING=mode))
        island = [f for f in _features(out) if _uid(f) == 25][0]
        assert island["gi_conf"] == 0, f"mode {mode}: island gi_conf={island['gi_conf']} (expected 0)"
        assert out.fields().lookupField("gi_class") >= 0, "no gi_class field"
        assert island["gi_class"] == "No neighbours", f"mode {mode}: island gi_class={island['gi_class']!r}"
        assert _is_null(island["gi_zscore"]), f"mode {mode}: island has a z-score {island['gi_zscore']}"


@check
def island_lisa_is_no_neighbours(tmp):
    layer = _island_layer()
    _res, out, _ctx, _fb = _run("local_moran_lisa", _common(layer, tmp, "lm_island"))
    island = [f for f in _features(out) if _uid(f) == 25][0]
    assert island["quadrant"] == "No neighbours", f"island quadrant={island['quadrant']!r}"
    assert island["lisa_nbrs"] == 0
    assert _is_null(island["lisa_p"]), f"island has a p-value {island['lisa_p']}"


def _null_nan_layer():
    return _layer("nulls", _grid(), [None, float("nan")] + BASE[2:])


def _donut_counts(html_text):
    """{label: count} from the donut legend ("Label: 12 (48.0%)")."""
    return {m.group(1).strip(): m.group(2) for m in re.finditer(r">([A-Za-z][^<:>]{1,40}): ([0-9.,]+) \(", html_text)}


@check
def null_and_nan_lisa_counts_match_layer(tmp):
    """P3/P11: NULL and NaN are 'No data', counted, drawn; report = layer."""
    layer = _null_nan_layer()
    report = _report(tmp, "lm_nulls")
    _res, out, _ctx, _fb = _run("local_moran_lisa", _common(layer, tmp, "lm_nulls", HTML_REPORT=report))
    feats = _features(out)
    assert len(feats) == 25, f"{len(feats)} features written"
    classes = [f["quadrant"] for f in feats]
    assert classes[0] == "No data" and classes[1] == "No data", f"NULL/NaN classes: {classes[:2]}"
    for f in feats[2:]:
        assert not _is_null(f["lisa_i"]), f"feature {_uid(f)} has no lisa_i"
        assert math.isfinite(float(f["lisa_i"])), f"feature {_uid(f)} lisa_i={f['lisa_i']}"
    layer_counts = {}
    for klass in classes:
        layer_counts[klass] = layer_counts.get(klass, 0) + 1
    donut = _donut_counts(Path(report).read_text(encoding="utf-8"))
    for klass, count in layer_counts.items():
        label = {"HH": "High-High (HH)", "LL": "Low-Low (LL)", "HL": "High-Low (HL) outlier",
                 "LH": "Low-High (LH) outlier"}.get(klass, klass)
        assert donut.get(label) == str(count), f"report says {label}={donut.get(label)}, layer has {count}"
    renderer = out.renderer()
    if renderer is not None and hasattr(renderer, "categories"):
        values = {c.value() for c in renderer.categories()}
        assert {"No data", "No neighbours"} <= values, f"renderer categories {values}"


@check
def weights_restandardised_after_dropping_nulls(tmp):
    """P5: a neighbour of a NULL feature uses weights re-standardised on valid rows."""
    import numpy as np

    layer = _null_nan_layer()
    _res, out, _ctx, _fb = _run("local_moran_lisa", _common(layer, tmp, "lm_p5", MULTIPLE_TESTING=0))
    feats = _features(out)
    values = {u: BASE[u] for u in range(2, 25)}
    uids = sorted(values)
    y = np.array([values[u] for u in uids], dtype=float)
    z = y - y.mean()
    m2 = float(np.sum(z ** 2) / len(z))
    index = {u: i for i, u in enumerate(uids)}

    def queen(u):
        r, c = divmod(u, 5)
        return [rr * 5 + cc for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                if (rr, cc) != (r, c) and 0 <= rr < 5 and 0 <= cc < 5]

    for u in (5, 6, 7):  # touch the NULL (0) and/or NaN (1) feature
        nbrs = [n for n in queen(u) if n in index]
        expected = z[index[u]] / m2 * float(np.mean([z[index[n]] for n in nbrs]))
        got = [f["lisa_i"] for f in feats if _uid(f) == u][0]
        assert abs(float(got) - expected) < 1e-9, f"uid {u}: lisa_i={got}, hand value {expected:.6f}"


@check
def null_and_nan_gi_star_shown_and_counted(tmp):
    layer = _null_nan_layer()
    report = _report(tmp, "gi_nulls")
    _res, out, _ctx, _fb = _run("getis_ord_gi", _common(layer, tmp, "gi_nulls", HTML_REPORT=report))
    feats = _features(out)
    assert len(feats) == 25
    nulls = [_uid(f) for f in feats if _is_null(f["gi_conf"])]
    assert not nulls, f"gi_conf NULL (hidden by the renderer) for {nulls}"
    assert feats[0]["gi_class"] == "No data" and feats[1]["gi_class"] == "No data"
    zs = [float(f["gi_zscore"]) for f in feats[2:]]
    assert all(math.isfinite(v) for v in zs) and any(abs(v) > 1e-9 for v in zs), "Gi* z-scores are all zero/NaN"
    layer_counts = {}
    for f in feats:
        layer_counts[f["gi_class"]] = layer_counts.get(f["gi_class"], 0) + 1
    donut = _donut_counts(Path(report).read_text(encoding="utf-8"))
    for klass, count in layer_counts.items():
        assert donut.get(klass) == str(count), f"report says {klass}={donut.get(klass)}, layer has {count}"


@check
def only_nan_values_fail_plainly(tmp):
    """P2/P3: a field with no finite value must not produce a 'result'."""
    layer = _layer("allnan", _grid(), [float("nan")] * 25)
    for alg in ("getis_ord_gi", "local_moran_lisa", "global_moran_autocorrelation"):
        params = _common(layer, tmp, f"allnan_{alg}")
        if alg == "global_moran_autocorrelation":
            params.pop("OUTPUT")
        _refused(alg, params, "finite")


@check
def other_cluster_outputs_show_missing_classes(tmp):
    """C2 beyond Gi*/LISA: NULL -> 'No data', island -> 'No neighbours', both drawn."""
    from qgis.core import QgsGeometry

    geometries = _grid() + [QgsGeometry.fromPolygonXY([_square(50000.0, 50000.0)])]
    values = [None] + BASE[1:] + [99.0]
    layer = _layer("mixed", geometries, values, extra={"w": [float((i * 3) % 7) for i in range(26)]})
    runs = (
        ("local_geary_c", "quadrant", _common(layer, tmp, "mix_lgc", PERMUTATIONS=99)),
        ("bivariate_lisa", "quadrant", dict(_common(layer, tmp, "mix_bil", PERMUTATIONS=99), FIELD_X="v", FIELD_Y="w")),
        ("losh_heteroskedasticity", "losh_type", _common(layer, tmp, "mix_losh")),
    )
    for alg, class_field, params in runs:
        _res, out, _ctx, _fb = _run(alg, params)
        by_uid = {_uid(f): f[class_field] for f in out.getFeatures()}
        assert by_uid.get(0) == "No data", f"{alg}: NULL feature class {by_uid.get(0)!r}"
        assert by_uid.get(25) == "No neighbours", f"{alg}: island class {by_uid.get(25)!r}"
        renderer = out.renderer()
        values_drawn = {c.value() for c in renderer.categories()} if hasattr(renderer, "categories") else set()
        assert {"No data", "No neighbours"} <= values_drawn, f"{alg}: renderer categories {values_drawn}"


@check
def nan_global_tools_run_without_traceback(tmp):
    """P3: Moran is computed on the finite rows; Geary no longer crashes in its chart."""
    layer = _null_nan_layer()
    report = _report(tmp, "gm_nan")
    _run("global_moran_autocorrelation", {"INPUT": layer, "FIELD": "v", "WEIGHT_TYPE": 0, "HTML_REPORT": report})
    text = Path(report).read_text(encoding="utf-8")
    assert not re.search(r">\s*-?nan\s*<", text), "Moran report shows nan"
    assert "Skipped: NaN or infinite value" in text, "Moran report does not count the NaN row"
    _run("geary_c_autocorrelation", {"INPUT": layer, "FIELD": "v", "WEIGHT_TYPE": 0, "PERMUTATIONS": 99,
                                     "HTML_REPORT": _report(tmp, "gc_nan")})


@check
def points_with_contiguity_refused_with_hint(tmp):
    """P2/P7: points + Queen gives every feature zero neighbours; refuse it."""
    layer = _points_layer()
    for alg in ("local_moran_lisa", "getis_ord_gi", "global_moran_autocorrelation", "geary_c_autocorrelation",
                "local_geary_c", "losh_heteroskedasticity"):
        params = _common(layer, tmp, f"pts_{alg}", PERMUTATIONS=99)
        if alg in ("global_moran_autocorrelation", "geary_c_autocorrelation"):
            params.pop("OUTPUT")
        _refused(alg, params, "K-Nearest", "Distance Band")


@check
def geographic_distance_band_refused(tmp):
    """P8: a distance band on EPSG:4326 is in degrees; refuse it with a hint."""
    layer = _points_layer(crs="EPSG:4326", step=0.01, x0=27.0, y0=38.0)
    for alg in ("local_moran_lisa", "getis_ord_gi", "global_moran_autocorrelation", "local_geary_c",
                "losh_heteroskedasticity"):
        params = _common(layer, tmp, f"geo_{alg}", WEIGHT_TYPE=3, DISTANCE_BAND=1000.0, PERMUTATIONS=99)
        if alg == "global_moran_autocorrelation":
            params.pop("OUTPUT")
        _refused(alg, params, "geographic", "reproject")
    _refused("general_g_autocorrelation", {"INPUT": layer, "FIELD": "v", "DISTANCE_BAND": 1000.0,
                                           "HTML_REPORT": _report(tmp, "geo_gg")}, "geographic", "reproject")


@check
def knn_stability_on_geographic_crs_runs_without_distance_variants(tmp):
    """P8: stability variants vary only the chosen family (no degree-valued bands)."""
    layer = _points_layer(crs="EPSG:4326", step=0.01, x0=27.0, y0=38.0)
    _res, out, _ctx, _fb = _run("getis_ord_gi", _common(layer, tmp, "gi_geo_knn", WEIGHT_TYPE=2,
                                                        STABILITY_ANALYSIS=True))
    runs = {f["gi_runs"] for f in out.getFeatures()}
    assert runs == {5}, f"gi_runs={runs} (expected base + 4 KNN variants)"


@check
def stability_skips_island_only_variants(tmp):
    """A variant that leaves every feature an island is skipped and named, not zero-filled."""
    layer = _points_layer()  # 100 m spacing
    report = _report(tmp, "gi_skip")
    _res, out, _ctx, _fb = _run("getis_ord_gi", _common(layer, tmp, "gi_skip", WEIGHT_TYPE=3, DISTANCE_BAND=105.0,
                                                        STABILITY_ANALYSIS=True, HTML_REPORT=report))
    runs = {f["gi_runs"] for f in out.getFeatures()}
    assert runs == {3}, f"gi_runs={runs} (expected base + the 1.1x and 1.25x bands; 0.75x and 0.9x skipped)"
    text = Path(report).read_text(encoding="utf-8")
    assert text.count("was skipped") >= 2, "report does not name the skipped stability runs"


@check
def ols_badge_says_not_computed_without_neighbours(tmp):
    """NaN residual Moran (no neighbours) must not read 'No Significant Autocorrelation'."""
    from qgis.core import QgsGeometry

    geometries = [QgsGeometry.fromPolygonXY([_square((i % 5) * 300.0, (i // 5) * 300.0)]) for i in range(25)]
    x2 = [float(i) for i in range(25)]
    layer = _layer("apart", geometries, [2.0 * v + ((v * 7) % 5) for v in x2], extra={"x2": x2})
    report = _report(tmp, "ols_apart")
    _run("ols_regression", {"INPUT": layer, "DEP_VAR": "v", "INDEPENDENTS": ["x2"],
                            "OUTPUT": "TEMPORARY_OUTPUT", "HTML_REPORT": report})
    text = Path(report).read_text(encoding="utf-8")
    assert "Not computed (no neighbours)" in text, "OLS badge does not say the residual Moran was not computed"
    assert "No Significant Autocorrelation" not in text, "OLS badge calls an uncomputed Moran 'no autocorrelation'"


@check
def consensus_legend_says_stable_not_confidence(tmp):
    """P12: the consensus map's legend is a vote, not a confidence level."""
    layer = _layer("gradient", _grid(6), [float(i % 6 + i // 6) for i in range(36)])
    _res, out, _ctx, _fb = _run("getis_ord_gi", _common(layer, tmp, "gi_cons", WEIGHT_TYPE=2, KNN=5,
                                                        STABILITY_ANALYSIS=True))
    renderer = out.renderer()
    assert hasattr(renderer, "categories"), f"renderer {type(renderer).__name__}"
    labels = [c.label() for c in renderer.categories()]
    assert {"Stable hot spot", "Stable cold spot", "Not stable"} <= set(labels), f"legend {labels}"
    assert not any("Confidence" in label for label in labels), f"legend still shows confidence: {labels}"
    classes = {f["gi_class"] for f in out.getFeatures()}
    assert classes <= {"Stable hot spot", "Stable cold spot", "Not stable", "No data", "No neighbours"}, classes


@check
def report_wording_and_integer_counts(tmp):
    """P14: corrections in words, counts as integers, help matches the code."""
    from qgis.core import QgsApplication

    layer = _layer("plain", _grid(), BASE)
    for alg, name in (("local_moran_lisa", "lm_words"), ("getis_ord_gi", "gi_words")):
        report = _report(tmp, name)
        _run(alg, _common(layer, tmp, name, MULTIPLE_TESTING=1, HTML_REPORT=report))
        text = Path(report).read_text(encoding="utf-8")
        assert "Benjamini-Hochberg false discovery rate" in text, f"{alg}: correction not in words"
        assert "fdr_bh" not in text, f"{alg}: report shows the code fdr_bh"
        assert not re.search(r": \d+\.00 \(", text), f"{alg}: counts printed with decimals"
    help_text = QgsApplication.processingRegistry().algorithmById(f"{PROVIDER}:local_moran_lisa").shortHelpString()
    assert "lisa_padj < 0.05" in help_text, "Local Moran help does not say classes use lisa_padj"


@check
def cancel_while_writing_leaves_no_partial_layer(tmp):
    """P10: a cancelled run must not hand back a half-written layer."""
    from qgis.core import QgsProcessingException

    layer = _layer("cancel", _grid(), BASE)
    for alg in ("local_moran_lisa", "getis_ord_gi"):
        try:
            res, out, _ctx, _fb = _run(alg, _common(layer, tmp, f"cancel_{alg}"), cancel_at_progress=60)
        except QgsProcessingException:
            continue
        n = out.featureCount() if out is not None else 0
        assert not res.get("OUTPUT") or out is None, f"{alg}: cancelled run returned a layer with {n} of 25 features"


@check
def input_attributes_survive_in_outputs(tmp):
    """3.12 wrote every input attribute as NULL (PyQGIS setFields clears values)."""
    layer = _layer("attrs", _grid(), BASE, extra={"x2": [float(i * i) for i in range(25)]})
    runs = (
        ("local_moran_lisa", _common(layer, tmp, "attr_lm")),
        ("getis_ord_gi", _common(layer, tmp, "attr_gi")),
        ("losh_heteroskedasticity", _common(layer, tmp, "attr_losh")),
        ("ols_regression", {"INPUT": layer, "DEP_VAR": "v", "INDEPENDENTS": ["x2"],
                            "OUTPUT": "TEMPORARY_OUTPUT", "HTML_REPORT": _report(tmp, "attr_ols")}),
    )
    for alg, params in runs:
        _res, out, _ctx, _fb = _run(alg, params)
        got = sorted((f["uid"], f["v"]) for f in out.getFeatures() if not _is_null(f["uid"]))
        assert got == [(i, BASE[i]) for i in range(25)], f"{alg}: input attributes lost ({len(got)} of 25 kept)"


@check
def hotspot_search_finds_the_tools(tmp):
    """P15: 'hotspot', 'hot spot', 'LISA' find Gi* and Local Moran by tags."""
    from qgis.core import QgsApplication

    provider = QgsApplication.processingRegistry().providerById(PROVIDER)
    tags = {alg.name(): [t.lower() for t in alg.tags()] for alg in provider.algorithms()}
    for word, expected in (("hotspot", {"getis_ord_gi", "local_moran_lisa"}),
                           ("hot spot", {"getis_ord_gi"}),
                           ("lisa", {"local_moran_lisa", "bivariate_lisa"}),
                           ("spatial autocorrelation", {"global_moran_autocorrelation"}),
                           ("local moran", {"local_moran_lisa"})):
        found = {name for name, values in tags.items() if word in values}
        assert expected <= found, f"tag {word!r} finds {sorted(found)}"


# ------------------------------------------------------------------- main --
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="folder that contains the planx_geostats package")
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(args.root.resolve()))

    from qgis.core import QgsApplication

    app = QgsApplication([], False)
    app.initQgis()
    sys.path.append(os.path.join(QgsApplication.prefixPath(), "python", "plugins"))
    import qgis
    sys.path.append(str(Path(qgis.__file__).resolve().parent.parent / "plugins"))
    from processing.core.Processing import Processing

    Processing.initialize()
    from planx_geostats.planx_geostats_provider import PlanXGeoStatsProvider

    provider = PlanXGeoStatsProvider()
    QgsApplication.processingRegistry().addProvider(provider)
    import planx_geostats
    print("plugin:", Path(planx_geostats.__file__).resolve().parent)

    failures = 0
    with tempfile.TemporaryDirectory(prefix="gs_robust_") as tmp:
        for func in CHECKS:
            if args.only and func.__name__ not in args.only:
                continue
            try:
                func(tmp)
                print(f"PASS {func.__name__}")
            except Exception as exc:  # report every check, then fail the run
                failures += 1
                detail = str(exc) if isinstance(exc, AssertionError) else traceback.format_exc(limit=4)
                print(f"FAIL {func.__name__}: {detail}".rstrip())
    print(f"GEOSTATS_ROBUSTNESS: {'PASS' if not failures else 'FAIL'} "
          f"({len(CHECKS) - failures if not args.only else '-'} passed, {failures} failed)")
    sys.stdout.flush()
    os._exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
