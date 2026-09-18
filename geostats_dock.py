# -*- coding: utf-8 -*-
"""PlanX GeoStats Lab dock: a docked Processing launcher.

A QTreeWidget grouped by the provider's 7 groups (each keeps its own
colour), one icon per tool, double-click to run, right-click for
documentation. Mirrors the sibling planx_urban_resilience plugin's
urban_resilience_dock.py. GeoStats has no shared GroupMixin across its
algorithm files (each already implements group()/groupId()/icon()
individually), so this module reads those directly off the registered
algorithms rather than off a per-file mixin constant. GEOSTATS_GROUPS
below must list every group id the provider's algorithms actually return -
a group missing here is silently omitted from the dock even though it is
still fully registered and runnable from the Processing Toolbox (the exact
bug that hid all of 06 | Machine Learning and Explainable AI until fixed).
"""
from __future__ import annotations

import os
import json

from qgis.PyQt.QtCore import Qt, QUrl, QSettings
from qgis.PyQt.QtGui import QBrush, QColor, QDesktopServices, QIcon, QPixmap
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qgis.core import QgsApplication

from .algorithms._mixins import DOC_BASE_URL

PLUGIN_DIR = os.path.dirname(__file__)
PROVIDER_ID = "planx_geostats"

#: (display_name, group_id, colour) in display order, matching each
#: algorithm's own group()/groupId() strings exactly.
GEOSTATS_GROUPS = (
    ("00 | Setup and Diagnostics", "planx_setup_diagnostics", "#64748B"),
    ("01 | Data Preparation and Neighborhoods", "planx_prepare_neighbors", "#0891B2"),
    ("02 | Urban Pattern Scan", "planx_pattern_scan", "#2563EB"),
    ("03 | Hot Spots and Spatial Outliers", "planx_hotspots_outliers", "#DC2626"),
    ("04 | Centers, Direction and Dispersion", "planx_center_direction_spread", "#7C3AED"),
    ("05 | Models and Scenarios", "planx_model_scenario", "#059669"),
    ("06 | Machine Learning and Explainable AI", "planx_ml_xai", "#4F46E5"),
    ("07 | Interpolation and Geostatistics", "planx_interpolation_geostatistics", "#0D9488"),
)

TOOL_PRESETS = {
    "planx_geostats:getis_ord_gi": {"MULTIPLE_TESTING": 1, "STABILITY_ANALYSIS": True},
    "planx_geostats:local_moran_lisa": {"MULTIPLE_TESTING": 1},
    "planx_geostats:local_geary_c": {"MULTIPLE_TESTING": 1},
    "planx_geostats:gwr_regression": {"BANDWIDTH_SELECTION": 1},
    "planx_geostats:calculate_distance_band": {"WEIGHT_TYPE": 2, "REPAIR_ISLANDS": True},
}

GUIDED_WORKFLOWS = {
    "Pattern → hot spots → outliers": (
        "planx_geostats:data_readiness_audit",
        "planx_geostats:global_moran_autocorrelation",
        "planx_geostats:getis_ord_gi",
        "planx_geostats:local_moran_lisa",
    ),
    "OLS → specification → spatial models": (
        "planx_geostats:ols_regression",
        "planx_geostats:lm_diagnostics",
        "planx_geostats:spatial_autoregression",
        "planx_geostats:spatial_error_regression",
    ),
    "Global → GWR → MGWR": (
        "planx_geostats:ols_regression",
        "planx_geostats:gwr_regression",
        "planx_geostats:multiscale_geographically_weighted_regression",
    ),
    "ML → spatial validation → explanation": (
        "planx_geostats:random_forest_regression",
        "planx_geostats:spatial_kfold_cv_evaluator",
        "planx_geostats:permutation_feature_importance",
        "planx_geostats:shap_global_importance",
    ),
}

# Every colour used below is pinned explicitly (never inherited from the
# host QPalette) - QGIS4/Qt6 hosts a dark palette by default and unpinned
# widgets render white-on-white, a bug the sibling urban_resilience dock
# already hit once in its old dashboard.py and had to fix.
_QSS = """
QWidget#geostatsDockBody { background: #f4f7fb; }
QLabel#geostatsHeader {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f2937, stop:1 #374151);
    color: white; font-weight: bold; font-size: 13px;
    padding: 10px 12px; border-radius: 6px;
}
QLabel#geostatsHint { color: #64748b; font-size: 11px; }
QLineEdit#geostatsSearch {
    background: #ffffff; color: #243044;
    border: 1px solid #cbd5e1; border-radius: 6px;
    padding: 5px 8px;
}
QLineEdit#geostatsSearch:focus { border: 1px solid #0f766e; }
QTreeWidget {
    background: #ffffff; color: #243044;
    border: 1px solid #d8e0ea; border-radius: 6px;
    font-size: 12px; outline: 0;
}
QTreeWidget::item { padding: 3px 2px; }
QTreeWidget::item:selected { background: #e6f4f2; color: #0f766e; }
QComboBox {
    background: #ffffff; color: #111827; border: 1px solid #cbd5e1;
    border-radius: 4px; padding: 5px 8px;
}
QPushButton {
    background: #e2e8f0; color: #111827; border: 1px solid #94a3b8;
    border-radius: 4px; padding: 5px 9px;
}
QPushButton:hover { background: #cbd5e1; }
QPushButton#geostatsDocBtn {
    background: #0f766e; color: white; font-weight: bold;
    border: none; border-radius: 4px; padding: 6px 12px;
}
QPushButton#geostatsDocBtn:hover { background: #0d9488; }
QToolTip {
    background: #102027; color: #ffffff;
    border: 1px solid #102027; padding: 4px 6px;
}
"""


def _swatch_icon(hex_color: str, size: int = 12) -> QIcon:
    """A small solid-colour square used as the group-header icon."""
    pix = QPixmap(size, size)
    pix.fill(QColor(hex_color))
    return QIcon(pix)


class GeoStatsDock(QDockWidget):
    """Browses the planx_geostats provider and launches algorithms."""

    def __init__(self, iface):
        super().__init__("PlanX GeoStats Lab")
        self.iface = iface
        self.setObjectName("PlanXGeoStatsDock")

        body = QWidget()
        body.setObjectName("geostatsDockBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("PlanX GeoStats Lab")
        header.setObjectName("geostatsHeader")
        layout.addWidget(header)

        hint = QLabel("Double-click a tool to run it. Right-click for documentation.")
        hint.setObjectName("geostatsHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        doc_row = QHBoxLayout()
        doc_btn = QPushButton("\U0001F4D6 Open Reference Manual")
        doc_btn.setObjectName("geostatsDocBtn")
        doc_btn.setToolTip("Open the comprehensive academic reference manual (GitHub Pages)")
        doc_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOC_BASE_URL)))
        doc_row.addWidget(doc_btn)
        layout.addLayout(doc_row)

        self.search = QLineEdit()
        self.search.setObjectName("geostatsSearch")
        self.search.setPlaceholderText("Search tools...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        workflow_row = QHBoxLayout()
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItems(GUIDED_WORKFLOWS)
        self.workflow_combo.setToolTip("Open each Processing step in a guided sequence")
        workflow_row.addWidget(self.workflow_combo, 1)
        workflow_button = QPushButton("Run workflow")
        workflow_button.clicked.connect(self._launch_workflow)
        workflow_row.addWidget(workflow_button)
        layout.addLayout(workflow_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.itemDoubleClicked.connect(self._launch)
        layout.addWidget(self.tree, 1)

        body.setStyleSheet(_QSS)
        self.setWidget(body)

        self._populate()

    # ------------------------------------------------------------------ #
    def _populate(self):
        self.tree.clear()
        provider = QgsApplication.processingRegistry().providerById(PROVIDER_ID)
        if provider is None:
            self.tree.addTopLevelItem(
                QTreeWidgetItem(["PlanX GeoStats Lab provider not loaded"]))
            return
        fallback = QIcon(os.path.join(PLUGIN_DIR, "icons", "icon.png"))
        self._add_shortcut_group(provider, "★ Favorites", self._stored_ids("favorites"), fallback)
        self._add_shortcut_group(provider, "↻ Recent", self._stored_ids("recent"), fallback)
        by_group_id = {}
        for alg in provider.algorithms():
            by_group_id.setdefault(alg.groupId(), []).append(alg)
        for display_name, group_id, color in GEOSTATS_GROUPS:
            algs = by_group_id.get(group_id, [])
            if not algs:
                continue
            parent = QTreeWidgetItem([display_name])
            parent.setIcon(0, _swatch_icon(color))
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            parent.setForeground(0, QBrush(QColor(color)))
            self.tree.addTopLevelItem(parent)
            for alg in sorted(algs, key=lambda a: a.displayName()):
                item = QTreeWidgetItem([alg.displayName()])
                try:
                    icon = alg.icon()
                except Exception:
                    icon = fallback
                item.setIcon(0, icon if not icon.isNull() else fallback)
                item.setToolTip(0, alg.shortHelpString())
                item.setData(0, Qt.ItemDataRole.UserRole, alg.id())
                parent.addChild(item)
            parent.setExpanded(True)

    def _settings(self):
        return QSettings()

    def _stored_ids(self, key):
        raw = self._settings().value(f"planx_geostats/{key}", "[]")
        try:
            return list(json.loads(str(raw)))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []

    def _save_ids(self, key, values):
        self._settings().setValue(f"planx_geostats/{key}", json.dumps(values[:20]))

    def _add_shortcut_group(self, provider, title, algorithm_ids, fallback):
        algorithms = {algorithm.id(): algorithm for algorithm in provider.algorithms()}
        available = [algorithms[algorithm_id] for algorithm_id in algorithm_ids if algorithm_id in algorithms]
        if not available:
            return
        parent = QTreeWidgetItem([title])
        font = parent.font(0)
        font.setBold(True)
        parent.setFont(0, font)
        self.tree.addTopLevelItem(parent)
        for algorithm in available:
            item = QTreeWidgetItem([algorithm.displayName()])
            icon = algorithm.icon()
            item.setIcon(0, icon if not icon.isNull() else fallback)
            item.setData(0, Qt.ItemDataRole.UserRole, algorithm.id())
            parent.addChild(item)
        parent.setExpanded(True)

    # ------------------------------------------------------------------ #
    def _launch(self, item, _column):
        alg_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not alg_id:
            return
        try:
            import processing
            recent = [value for value in self._stored_ids("recent") if value != alg_id]
            self._save_ids("recent", [alg_id] + recent)
            processing.execAlgorithmDialog(alg_id, dict(TOOL_PRESETS.get(alg_id, {})))
            self._populate()
        except Exception as exc:
            self.iface.messageBar().pushWarning(
                "PlanX GeoStats Lab", f"Could not open tool: {exc}")

    def _context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item:
            return
        alg_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not alg_id:
            return
        alg = QgsApplication.processingRegistry().algorithmById(alg_id)
        if alg is None:
            return
        menu = QMenu(self)
        favorites = self._stored_ids("favorites")
        favorite_action = menu.addAction("Remove from Favorites" if alg_id in favorites else "Add to Favorites")
        favorite_action.triggered.connect(lambda: self._toggle_favorite(alg_id))
        if alg_id in TOOL_PRESETS:
            preset_action = menu.addAction("Launch with Recommended Preset")
            preset_action.triggered.connect(lambda: self._launch_with_preset(alg_id))
        help_action = menu.addAction("View Documentation")
        help_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(alg.helpUrl())))
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def _toggle_favorite(self, algorithm_id):
        favorites = self._stored_ids("favorites")
        if algorithm_id in favorites:
            favorites.remove(algorithm_id)
        else:
            favorites.insert(0, algorithm_id)
        self._save_ids("favorites", favorites)
        self._populate()

    def _launch_with_preset(self, algorithm_id):
        import processing
        processing.execAlgorithmDialog(algorithm_id, dict(TOOL_PRESETS.get(algorithm_id, {})))

    def _launch_workflow(self):
        import processing
        workflow = self.workflow_combo.currentText()
        active_layer = self.iface.activeLayer()
        for index, algorithm_id in enumerate(GUIDED_WORKFLOWS.get(workflow, ())):
            parameters = dict(TOOL_PRESETS.get(algorithm_id, {}))
            if active_layer is not None:
                parameters["INPUT"] = active_layer
            self.iface.messageBar().pushInfo(
                "PlanX GeoStats Lab",
                f"Workflow step {index + 1}/{len(GUIDED_WORKFLOWS[workflow])}: {algorithm_id.split(':')[-1]}",
            )
            result = processing.execAlgorithmDialog(algorithm_id, parameters)
            if result is None:
                break

    # ------------------------------------------------------------------ #
    def _filter(self, text: str):
        needle = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(i)
            any_visible = False
            for j in range(group_item.childCount()):
                child = group_item.child(j)
                visible = not needle or needle in child.text(0).lower()
                child.setHidden(not visible)
                any_visible = any_visible or visible
            group_item.setHidden(not any_visible)
            if needle:
                group_item.setExpanded(True)
