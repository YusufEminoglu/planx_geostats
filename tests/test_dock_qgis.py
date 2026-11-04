#!/usr/bin/env python3
r"""GeoStats Lab dock checks inside a real (offscreen) QGIS.

Covers the 3.13 dock criteria:
  * every dock text, tree rows and group headers included, reaches a 4.5:1
    contrast ratio under the default UI theme, Night Mapping and Blend of
    Gray, measured on widget.grab() pixels (P13);
  * double-click opens a tool with the Processing Toolbox defaults, and the
    optional preset is only offered under a label that names every change
    (P12);
  * guided-workflow messages use tool display names, the manual tooltip
    names geophilo.com (P14);
  * the dock search finds tools by common words: hotspot, hot spot, LISA,
    cluster, autocorrelation, moran (P15).

Run with a QGIS Python runtime:
  C:\OSGeo4W\bin\python-qgis-ltr.bat planx_geostats\tests\test_dock_qgis.py --root C:\Users\YE\PyCharmMiscProject\qgis_plugins
  C:\OSGeo4W\bin\python-qgis.bat planx_geostats\tests\test_dock_qgis.py --root C:\Users\YE\PyCharmMiscProject\qgis_plugins
A fresh QGIS profile directory is used unless QGIS_CUSTOM_CONFIG_PATH is set.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback
from collections import Counter

sys.dont_write_bytecode = True
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROVIDER_ID = "planx_geostats"
THEMES = ("default", "Night Mapping", "Blend of Gray")
MIN_CONTRAST = 4.5

#: query -> algorithm names that must stay visible after typing it.
SEARCH_EXPECTATIONS = {
    "hotspot": {"getis_ord_gi"},
    "hot spot": {"getis_ord_gi"},
    "LISA": {"local_moran_lisa", "bivariate_lisa", "eb_rate_lisa"},
    "cluster": {"local_moran_lisa", "getis_ord_gi", "multivariate_clustering"},
    "autocorrelation": {"global_moran_autocorrelation", "local_moran_lisa", "geary_c_autocorrelation"},
    "moran": {"global_moran_autocorrelation", "local_moran_lisa", "eb_rate_lisa", "local_partial_moran"},
}

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


# --------------------------------------------------------------------------- #
# contrast from pixels
def _luminance(rgb):
    channels = []
    for value in rgb:
        x = value / 255.0
        channels.append(x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _ratio(a, b):
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def text_contrast(with_text, without_text, x0, y0, x1, y1):
    """(ratio, background, text) for one text area.

    ``with_text`` and ``without_text`` are grabs of the same widget with the
    text shown and blanked: pixels that differ are the text, the commonest
    colour of the blank grab is its background, and the ratio is taken for
    the text pixel farthest from that background (a glyph stroke). No
    differing pixel at all means no visible text: ratio 0."""
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(with_text.width(), without_text.width(), x1)
    y1 = min(with_text.height(), without_text.height(), y1)
    background_pixels, text_pixels = [], set()
    for y in range(y0, y1):
        for x in range(x0, x1):
            a, b = with_text.pixelColor(x, y), without_text.pixelColor(x, y)
            rgb_b = (b.red(), b.green(), b.blue())
            background_pixels.append(rgb_b)
            rgb_a = (a.red(), a.green(), a.blue())
            if rgb_a != rgb_b:
                text_pixels.add(rgb_a)
    if not background_pixels:
        return 0.0, None, None
    background = Counter(background_pixels).most_common(1)[0][0]
    if not text_pixels:
        return 0.0, background, None
    text = max(text_pixels, key=lambda p: _ratio(p, background))
    return _ratio(text, background), background, text


def _hex(rgb):
    return "#%02x%02x%02x" % rgb if rgb else "-"


# --------------------------------------------------------------------------- #
class _Bar:
    def __init__(self):
        self.messages = []

    def pushInfo(self, title, text):
        self.messages.append(text)

    def pushWarning(self, title, text):
        self.messages.append(text)


class _Iface:
    def __init__(self):
        self.bar = _Bar()

    def messageBar(self):
        return self.bar

    def activeLayer(self):
        return None


def _tree_rows(tree):
    rows = []
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        rows.append(top)
        for j in range(top.childCount()):
            rows.append(top.child(j))
    return rows


def _find_item(dock, name):
    from qgis.PyQt.QtCore import Qt
    for item in _tree_rows(dock.tree):
        alg_id = item.data(0, Qt.ItemDataRole.UserRole)
        if alg_id == f"{PROVIDER_ID}:{name}":
            return item
    return None


def contrast_checks(app, gd, theme):
    from qgis.PyQt.QtCore import QPoint
    from qgis.PyQt.QtWidgets import QComboBox, QLabel, QLineEdit, QPushButton
    dock = gd.GeoStatsDock(_Iface())
    dock.resize(420, 2900)
    dock.show()
    app.processEvents()
    tree = dock.tree
    rows = _tree_rows(tree)

    def tree_pair():
        shown = tree.viewport().grab().toImage()
        texts = [item.text(0) for item in rows]
        for item in rows:
            item.setText(0, "")
        app.processEvents()
        blank = tree.viewport().grab().toImage()
        for item, text in zip(rows, texts):
            item.setText(0, text)
        app.processEvents()
        return shown, blank

    shown, blank = tree_pair()
    worst = (99.0, "", None, None)
    measured = 0
    for item in rows:
        rect = tree.visualItemRect(item)
        if rect.isEmpty() or rect.bottom() >= shown.height():
            continue
        ratio, bg, fg = text_contrast(shown, blank, rect.left(), rect.top(), rect.right() + 1, rect.bottom() + 1)
        measured += 1
        if ratio < worst[0]:
            worst = (ratio, item.text(0), bg, fg)
    check(f"[{theme}] tree text contrast >= {MIN_CONTRAST}:1 ({measured} rows)",
          measured > 50 and worst[0] >= MIN_CONTRAST,
          f"worst {worst[0]:.2f} '{worst[1][:40]}' text {_hex(worst[3])} on {_hex(worst[2])}")

    # a selected tool row and a selected group header
    for name in ("getis_ord_gi", None):
        item = _find_item(dock, name) if name else tree.topLevelItem(tree.topLevelItemCount() - 1)
        if item is None:
            check(f"[{theme}] selected row {name} present", False)
            continue
        tree.clearSelection()
        tree.setCurrentItem(item)
        item.setSelected(True)
        app.processEvents()
        shown, blank = tree_pair()
        rect = tree.visualItemRect(item)
        ratio, bg, fg = text_contrast(shown, blank, rect.left(), rect.top(), rect.right() + 1, rect.bottom() + 1)
        check(f"[{theme}] selected row '{item.text(0)[:30]}' contrast >= {MIN_CONTRAST}:1",
              ratio >= MIN_CONTRAST, f"{ratio:.2f} text {_hex(fg)} on {_hex(bg)}")
    tree.clearSelection()

    # every other text in the dock body, measured in place on the body grab
    body = dock.widget()
    targets = []
    for label in body.findChildren(QLabel):
        targets.append((f"label '{label.text()[:30]}'", label, lambda w=label: w.setText("")))
    for edit in body.findChildren(QLineEdit):
        edit.clear()
        targets.append(("search box placeholder", edit, lambda w=edit: w.setPlaceholderText("")))
    for combo in body.findChildren(QComboBox):
        targets.append(("workflow combo", combo,
                        lambda w=combo: w.setItemText(w.currentIndex(), "")))
    for button in body.findChildren(QPushButton):
        targets.append((f"button '{button.text()[:30]}'", button, lambda w=button: w.setText("")))
    targets = [t for t in targets if t[1].isVisible()]
    app.processEvents()
    for _label, widget, _blank in targets:
        widget.setFixedSize(widget.size())
    app.processEvents()
    shown = body.grab().toImage()
    for _label, _widget, blank_text in targets:
        blank_text()
    app.processEvents()
    blank = body.grab().toImage()
    for label, widget, _blank in targets:
        origin = widget.mapTo(body, QPoint(0, 0))
        ratio, bg, fg = text_contrast(shown, blank, origin.x(), origin.y(),
                                      origin.x() + widget.width(), origin.y() + widget.height())
        check(f"[{theme}] {label} contrast >= {MIN_CONTRAST}:1", ratio >= MIN_CONTRAST,
              f"{ratio:.2f} text {_hex(fg)} on {_hex(bg)}")
    dock.close()
    dock.deleteLater()
    app.processEvents()


def behaviour_checks(app, gd, registry):
    import processing
    from qgis.PyQt.QtWidgets import QPushButton

    calls = []

    def fake_dialog(alg_id, params=None, *args, **kwargs):
        calls.append((alg_id, dict(params or {})))
        return {}

    original = processing.execAlgorithmDialog
    processing.execAlgorithmDialog = fake_dialog
    try:
        iface = _Iface()
        dock = gd.GeoStatsDock(iface)

        # P12: double-click == Processing Toolbox defaults
        for name in ("getis_ord_gi", "local_moran_lisa", "local_geary_c", "gwr_regression",
                     "calculate_distance_band"):
            item = _find_item(dock, name)
            if item is None:
                check(f"double-click {name}: tool in dock", False)
                continue
            calls.clear()
            dock._launch(item, 0)
            alg = registry.algorithmById(f"{PROVIDER_ID}:{name}")
            changed = {}
            for key, value in (calls[0][1] if calls else {}).items():
                definition = alg.parameterDefinition(key)
                if definition is None or definition.defaultValue() != value:
                    changed[key] = value
            check(f"double-click {name} opens Toolbox defaults", calls and not changed,
                  f"changed {changed}" if changed else "")

        # P12: a preset, if offered, names every parameter it changes
        item = _find_item(dock, "getis_ord_gi")
        alg = registry.algorithmById(f"{PROVIDER_ID}:getis_ord_gi")
        try:
            menu = dock._build_context_menu(item)
            texts = [a.text() for a in menu.actions()]
            preset_texts = [t for t in texts if "preset" in t.lower()]
            missing = []
            for t in preset_texts:
                for key in gd.TOOL_PRESETS.get(alg.id(), {}):
                    if alg.parameterDefinition(key).description() not in t:
                        missing.append(key)
            check("preset menu entry names every changed parameter", not missing,
                  f"entries {preset_texts}; unnamed {missing}")
        except Exception as exc:
            check("preset menu entry names every changed parameter", False, repr(exc)[:160])

        # P14: workflow messages use display names; steps open with Toolbox defaults
        bad = []
        for index in range(dock.workflow_combo.count()):
            dock.workflow_combo.setCurrentIndex(index)
            iface.bar.messages.clear()
            calls.clear()
            dock._launch_workflow()
            steps = gd.GUIDED_WORKFLOWS[dock.workflow_combo.currentText()]
            for step, message in zip(steps, iface.bar.messages):
                step_alg = registry.algorithmById(step)
                if step_alg.displayName() not in message or step_alg.name() in message:
                    bad.append(message)
            for alg_id, params in calls:
                extra = {k: v for k, v in params.items() if k != "INPUT"}
                if extra:
                    bad.append(f"{alg_id} preset {extra}")
            if len(iface.bar.messages) != len(steps):
                bad.append(f"{len(iface.bar.messages)} messages for {len(steps)} steps")
        check("workflow messages show tool display names, no hidden presets", not bad, "; ".join(bad[:3]))

        # P14: manual tooltip names geophilo.com
        button = dock.widget().findChild(QPushButton, "geostatsDocBtn")
        tip = button.toolTip() if button is not None else ""
        check("manual button tooltip says geophilo.com", "geophilo.com" in tip and "GitHub" not in tip, tip)

        # P15: search by common words
        from qgis.PyQt.QtCore import Qt
        for query, expected in SEARCH_EXPECTATIONS.items():
            dock.search.setText(query)
            app.processEvents()
            visible = set()
            for top_index in range(dock.tree.topLevelItemCount()):
                top = dock.tree.topLevelItem(top_index)
                for j in range(top.childCount()):
                    child = top.child(j)
                    if not child.isHidden() and not top.isHidden():
                        visible.add(str(child.data(0, Qt.ItemDataRole.UserRole)).split(":")[-1])
            missing = expected - visible
            check(f"search '{query}' finds {sorted(expected)}", not missing, f"missing {sorted(missing)}")
        dock.search.setText("zzzz-no-such-tool")
        app.processEvents()
        hidden_all = all(dock.tree.topLevelItem(i).isHidden() for i in range(dock.tree.topLevelItemCount()))
        check("search with no match hides every group", hidden_all)
        dock.search.setText("")
        dock.close()
    finally:
        processing.execAlgorithmDialog = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        help="directory that contains the planx_geostats package")
    args = parser.parse_args()
    scratch = []
    if not os.environ.get("QGIS_CUSTOM_CONFIG_PATH"):
        scratch.append(tempfile.mkdtemp(prefix="geostats_dock_profile_"))
        os.environ["QGIS_CUSTOM_CONFIG_PATH"] = scratch[-1]
    if not os.environ.get("QT_QPA_FONTDIR") and os.name == "nt":
        # The offscreen platform reads fonts from QT_QPA_FONTDIR only; without
        # real fonts it draws no glyphs and there is no text to measure.
        system_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        font_dir = tempfile.mkdtemp(prefix="geostats_dock_fonts_")
        scratch.append(font_dir)
        for font in ("arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf"):
            if os.path.exists(os.path.join(system_fonts, font)):
                shutil.copy(os.path.join(system_fonts, font), font_dir)
        os.environ["QT_QPA_FONTDIR"] = font_dir
    sys.path.insert(0, os.path.abspath(args.root))

    from qgis.core import Qgis, QgsApplication
    app = QgsApplication([], True)
    app.initQgis()
    sys.path.append(os.path.join(QgsApplication.prefixPath(), "python", "plugins"))
    from processing.core.Processing import Processing
    Processing.initialize()
    from planx_geostats.planx_geostats_provider import PlanXGeoStatsProvider
    from planx_geostats import geostats_dock as gd
    print(f"QGIS {Qgis.QGIS_VERSION}; dock module {gd.__file__}", flush=True)
    registry = QgsApplication.processingRegistry()
    provider = PlanXGeoStatsProvider()
    registry.addProvider(provider)

    try:
        behaviour_checks(app, gd, registry)
    except Exception:
        check("behaviour checks ran", False, traceback.format_exc()[-400:])
    available = set(QgsApplication.uiThemes().keys())
    for theme in THEMES:
        if theme not in available:
            check(f"[{theme}] UI theme available", False, f"have {sorted(available)}")
            continue
        QgsApplication.setUITheme(theme)
        app.processEvents()
        try:
            contrast_checks(app, gd, theme)
        except Exception:
            check(f"[{theme}] contrast checks ran", False, traceback.format_exc()[-400:])
    QgsApplication.setUITheme("default")

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed", flush=True)
    sys.stdout.flush()
    for path in scratch:
        shutil.rmtree(path, ignore_errors=True)
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
