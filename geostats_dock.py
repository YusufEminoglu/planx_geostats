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

import contextlib
import os
import json
import re

from qgis.PyQt.QtCore import QEvent, Qt, QUrl, QSettings
from qgis.PyQt.QtGui import QBrush, QColor, QDesktopServices, QIcon, QPalette, QPixmap
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDockWidget,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
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
    ("00 | Setup and Diagnostics", "planx_setup_diagnostics", "#475569"),
    ("01 | Data Preparation and Neighborhoods", "planx_prepare_neighbors", "#0E7490"),
    ("02 | Urban Pattern Scan", "planx_pattern_scan", "#1D4ED8"),
    ("03 | Hot Spots and Spatial Outliers", "planx_hotspots_outliers", "#B91C1C"),
    ("04 | Centers, Direction and Dispersion", "planx_center_direction_spread", "#6D28D9"),
    ("05 | Models and Scenarios", "planx_model_scenario", "#047857"),
    ("06 | Machine Learning and Explainable AI", "planx_ml_xai", "#4338CA"),
    ("07 | Interpolation and Geostatistics", "planx_interpolation_geostatistics", "#0F766E"),
)

#: Optional, explicitly named presets. They are applied only from the
#: right-click "Launch with preset: ..." action, whose label spells out every
#: parameter it changes. Double-click and the guided workflows open each tool
#: with exactly the defaults the Processing Toolbox shows - no hidden preset.
TOOL_PRESETS = {
    "planx_geostats:getis_ord_gi": {"MULTIPLE_TESTING": 1, "STABILITY_ANALYSIS": True},
    "planx_geostats:local_moran_lisa": {"MULTIPLE_TESTING": 1},
    "planx_geostats:local_geary_c": {"MULTIPLE_TESTING": 1},
    "planx_geostats:gwr_regression": {"BANDWIDTH_SELECTION": 1},
    "planx_geostats:calculate_distance_band": {"WEIGHT_TYPE": 2, "REPAIR_ISLANDS": True},
}

#: Extra search words per algorithm name, matched by the dock search box in
#: addition to the display name and the algorithm's own tags(). Covers the
#: words people actually type (hotspot, LISA, cluster, autocorrelation ...).
SEARCH_KEYWORDS = {
    "getis_ord_gi": "hot spot hotspot cold spot coldspot gi* local g cluster "
                    "local spatial autocorrelation fdr",
    "local_moran_lisa": "lisa local moran cluster outlier hot spot hotspot "
                        "spatial autocorrelation local spatial autocorrelation geoda pysal esda",
    "global_moran_autocorrelation": "moran global moran spatial autocorrelation clustering "
                                    "geoda pysal esda",
    "general_g_autocorrelation": "general g hot spot hotspot clustering spatial autocorrelation",
    "geary_c_autocorrelation": "geary spatial autocorrelation",
    "incremental_spatial_autocorrelation": "moran distance band spatial autocorrelation",
    "local_geary_c": "lisa local geary cluster outlier spatial autocorrelation",
    "bivariate_lisa": "lisa bivariate moran cluster outlier spatial autocorrelation",
    "eb_rate_lisa": "lisa moran empirical bayes rate cluster outlier",
    "local_partial_moran": "lisa moran cluster",
    "multivariate_local_geary": "lisa geary cluster",
    "join_count_statistics": "spatial autocorrelation binary",
    "local_join_counts": "lisa cluster binary spatial autocorrelation",
    "losh_heteroskedasticity": "hot spot hotspot local variance",
    "gamma_index": "spatial autocorrelation",
    "global_bivariate_lee_l": "bivariate spatial association spatial autocorrelation",
    "bivariate_spatial_association_lees_l": "bivariate spatial association local",
    "model_residual_autocorrelation": "moran residual spatial autocorrelation",
    "calculate_distance_band": "spatial weights queen rook knn k nearest neighbours "
                               "neighbors distance band islands",
    "skater_regionalization": "skater regionalization regions zones cluster",
    "max_p_regions": "max-p maxp regionalization regions zones cluster",
    "azp_regionalization": "azp regionalization regions zones cluster",
    "redcap_regionalization": "redcap regionalization regions zones cluster",
    "regionalization_quality": "regionalization regions zones cluster",
    "multivariate_clustering": "kmeans k-means cluster",
    "dbscan_clustering": "cluster density",
    "hdbscan_clustering": "cluster density",
    "gmm_clustering": "cluster gmm",
    "spatial_autoregression": "sar spatial lag",
    "spatial_error_regression": "sem",
    "ols_regression": "linear regression",
    "geodetector_q_statistic": "stratified heterogeneity",
}


def _normalize(text: str) -> str:
    """Lower-case and drop spaces, hyphens, underscores and apostrophes so
    'hot spot', 'hot-spot' and 'hotspot' all compare equal."""
    return re.sub(r"[\s\-_'’]+", "", str(text).lower())


def search_haystack(alg) -> str:
    """Normalized text the dock search matches for one algorithm."""
    try:
        tags = " ".join(alg.tags())
    except Exception:
        tags = ""
    extra = SEARCH_KEYWORDS.get(alg.name(), "")
    return _normalize(" ".join((alg.displayName(), tags, extra)))


def matches_search(needle: str, haystack: str) -> bool:
    """True when the whole query, or every word of it, occurs in haystack."""
    if not needle.strip():
        return True
    if _normalize(needle) in haystack:
        return True
    words = [_normalize(word) for word in needle.split()]
    words = [word for word in words if word]
    return bool(words) and all(word in haystack for word in words)


def preset_summary(alg_id: str) -> str:
    """Plain words for what a TOOL_PRESETS entry changes, e.g.
    'Multiple testing correction = FDR (Benjamini-Hochberg); Stability analysis = on'."""
    preset = TOOL_PRESETS.get(alg_id, {})
    if not preset:
        return ""
    alg = QgsApplication.processingRegistry().algorithmById(alg_id)
    parts = []
    for key, value in preset.items():
        definition = alg.parameterDefinition(key) if alg is not None else None
        label = definition.description() if definition is not None else key
        shown = value
        if isinstance(value, bool):
            shown = "on" if value else "off"
        elif definition is not None and hasattr(definition, "options"):
            with contextlib.suppress(Exception):
                options = definition.options()
                if isinstance(value, int) and 0 <= value < len(options):
                    shown = options[value]
        parts.append(f"{label} = {shown}")
    return "; ".join(parts)


def algorithm_display_name(alg_id: str) -> str:
    """The tool's display name, falling back to its id only if unregistered."""
    alg = QgsApplication.processingRegistry().algorithmById(alg_id)
    return alg.displayName() if alg is not None else alg_id.split(":")[-1]


GUIDED_WORKFLOWS = {
    "Pattern → hot spots → outliers": (
        "planx_geostats:data_readiness_audit",
        "planx_geostats:global_moran_autocorrelation",
        "planx_geostats:getis_ord_gi",
        "planx_geostats:local_moran_lisa",
    ),
    "Rates → smoothing → EB clusters": (
        "planx_geostats:rate_smoothing",
        "planx_geostats:eb_rate_lisa",
        "planx_geostats:losh_heteroskedasticity",
    ),
    "Clusters → covariates → what remains": (
        "planx_geostats:local_moran_lisa",
        "planx_geostats:local_partial_moran",
        "planx_geostats:multivariate_local_geary",
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

# QGIS UI themes (Night Mapping, Blend of Gray) set QTreeView::item { color },
# which beats the QTreeWidget colour, so ::item colour is pinned too.
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
QLabel#geostatsHint { color: #475569; font-size: 11px; }
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
QTreeWidget::item, QTreeWidget::branch { background: transparent; }
QTreeWidget::item { color: #243044; padding: 3px 2px; }
QTreeWidget::item:disabled { color: #475569; }
QTreeWidget::item:selected, QTreeWidget::branch:selected {
    background: #e6f4f2; color: #115e59;
}
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


#: Item data role holding a group header's text colour (painted by
#: _GroupColourDelegate, because a QSS ``::item { color }`` - ours or a UI
#: theme's - overrides QTreeWidgetItem.setForeground()).
GROUP_COLOUR_ROLE = int(Qt.ItemDataRole.UserRole) + 1
#: Item data role holding the normalized search text (name, tags, keywords).
SEARCH_ROLE = int(Qt.ItemDataRole.UserRole) + 2


class _SearchLineEdit(QLineEdit):
    """Search box whose placeholder keeps a 4.5:1 colour.

    A style sheet ``color`` makes Qt draw the placeholder at half alpha
    (#243044 becomes about 2.9:1 on white) and every re-polish resets the
    palette, so the placeholder colour is re-applied after each polish."""

    PLACEHOLDER = QColor("#475569")

    def event(self, event):
        result = super().event(event)
        if event.type() in (QEvent.Type.Polish, QEvent.Type.StyleChange, QEvent.Type.PaletteChange):
            palette = self.palette()
            if palette.color(QPalette.ColorRole.PlaceholderText) != self.PLACEHOLDER:
                palette.setColor(QPalette.ColorRole.PlaceholderText, self.PLACEHOLDER)
                self.setPalette(palette)
        return result


class _GroupColourDelegate(QStyledItemDelegate):
    """Draws group-header text in the group's own colour on any theme."""

    def paint(self, painter, option, index):
        colour = index.data(GROUP_COLOUR_ROLE)
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        if not colour or selected:
            super().paint(painter, option, index)
            return
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()
        text = opt.text
        text_rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, widget)
        opt.text = ""
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        painter.save()
        painter.setFont(opt.font)
        painter.setPen(QColor(colour))
        elided = opt.fontMetrics.elidedText(text, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), elided)
        painter.restore()


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
        doc_btn.setToolTip("Open the reference manual on geophilo.com")
        self.doc_btn = doc_btn
        doc_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOC_BASE_URL)))
        doc_row.addWidget(doc_btn)
        layout.addLayout(doc_row)

        self.search = _SearchLineEdit()
        self.search.setObjectName("geostatsSearch")
        self.search.setPlaceholderText("Search tools (e.g. hotspot, LISA, moran)...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        workflow_row = QHBoxLayout()
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItems(GUIDED_WORKFLOWS)
        self.workflow_combo.setToolTip("Open each Processing step in a guided sequence")
        workflow_row.addWidget(self.workflow_combo, 1)
        workflow_button = QPushButton("Run workflow")
        self.workflow_button = workflow_button
        workflow_button.clicked.connect(self._launch_workflow)
        workflow_row.addWidget(workflow_button)
        layout.addLayout(workflow_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setItemDelegate(_GroupColourDelegate(self.tree))
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
            parent.setData(0, GROUP_COLOUR_ROLE, color)
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
                item.setData(0, SEARCH_ROLE, search_haystack(alg))
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
            item.setData(0, SEARCH_ROLE, search_haystack(algorithm))
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
            # Same defaults as the Processing Toolbox: no preset on double-click.
            processing.execAlgorithmDialog(alg_id, {})
            self._populate()
        except Exception as exc:
            self.iface.messageBar().pushWarning(
                "PlanX GeoStats Lab", f"Could not open tool: {exc}")

    def _context_menu(self, pos):
        item = self.tree.itemAt(pos)
        menu = self._build_context_menu(item) if item else None
        if menu is not None:
            menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def _build_context_menu(self, item):
        alg_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not alg_id:
            return None
        alg = QgsApplication.processingRegistry().algorithmById(alg_id)
        if alg is None:
            return None
        menu = QMenu(self)
        favorites = self._stored_ids("favorites")
        favorite_action = menu.addAction("Remove from Favorites" if alg_id in favorites else "Add to Favorites")
        favorite_action.triggered.connect(lambda: self._toggle_favorite(alg_id))
        if alg_id in TOOL_PRESETS:
            preset_action = menu.addAction(f"Launch with preset: {preset_summary(alg_id)}")
            preset_action.triggered.connect(lambda: self._launch_with_preset(alg_id))
        help_action = menu.addAction("View Documentation")
        help_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(alg.helpUrl())))
        return menu

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
        self.iface.messageBar().pushInfo(
            "PlanX GeoStats Lab",
            f"{algorithm_display_name(algorithm_id)} opened with a preset: "
            f"{preset_summary(algorithm_id)}. The Processing Toolbox opens it without these changes.",
        )
        processing.execAlgorithmDialog(algorithm_id, dict(TOOL_PRESETS.get(algorithm_id, {})))

    def _launch_workflow(self):
        import processing
        workflow = self.workflow_combo.currentText()
        active_layer = self.iface.activeLayer()
        steps = GUIDED_WORKFLOWS.get(workflow, ())
        for index, algorithm_id in enumerate(steps):
            # Toolbox defaults; only the active layer is filled in.
            parameters = {}
            if active_layer is not None:
                parameters["INPUT"] = active_layer
            self.iface.messageBar().pushInfo(
                "PlanX GeoStats Lab",
                f"Workflow step {index + 1}/{len(steps)}: {algorithm_display_name(algorithm_id)}",
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
                haystack = child.data(0, SEARCH_ROLE) or _normalize(child.text(0))
                visible = matches_search(needle, haystack)
                child.setHidden(not visible)
                any_visible = any_visible or visible
            group_item.setHidden(not any_visible)
            if needle:
                group_item.setExpanded(True)
