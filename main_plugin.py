# -*- coding: utf-8 -*-
"""PlanX GeoStats Lab spatial statistics suite for QGIS.

Registers the PlanX GeoStats Lab provider. All user-facing tools appear under
the Processing Toolbox, including setup and dependency diagnostics.
"""
from __future__ import annotations

from typing import Optional

from qgis.core import QgsApplication
from qgis.PyQt.QtCore import Qt

from .geostats_dock import GeoStatsDock
from .planx_geostats_provider import PlanXGeoStatsProvider


class PlanXGeoStatsPlugin:
    DIAGNOSTICS_PATH = "PlanX GeoStats Lab > 00 | Setup and Diagnostics"

    def __init__(self, iface):
        self.iface = iface
        self.provider: Optional[PlanXGeoStatsProvider] = None
        self.dock: Optional[GeoStatsDock] = None
        self.action = None

    def initGui(self) -> None:
        self.provider = PlanXGeoStatsProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)
        self._warn_if_dependencies_missing()

        self.dock = GeoStatsDock(self.iface)
        self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        # toggleViewAction() is the dock's own checkable show/hide action -
        # clicking it again hides the panel, and it stays in sync if the
        # user closes the dock via its own [x] button too (same behaviour
        # as every native QGIS panel toggle icon).
        self.action = self.dock.toggleViewAction()
        self.action.setIcon(self.provider.icon())
        self.action.setToolTip("Show or hide the PlanX GeoStats Lab panel")
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&PlanX GeoStats Lab", self.action)

    def unload(self) -> None:
        if self.action is not None:
            self.iface.removePluginMenu("&PlanX GeoStats Lab", self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None
        if self.dock is not None:
            self.iface.removeDockWidget(self.dock)
            self.dock = None
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None

    def _warn_if_dependencies_missing(self) -> None:
        try:
            from . import DEPENDENCIES_MISSING, MISSING_LIBS
        except Exception:
            return
        if not DEPENDENCIES_MISSING:
            return
        missing = ", ".join(MISSING_LIBS)
        self.iface.messageBar().pushWarning(
            "PlanX GeoStats Lab",
            f"Optional GeoStats libraries are missing: {missing}. "
            f"Run {self.DIAGNOSTICS_PATH} tools to review and install them.",
        )
