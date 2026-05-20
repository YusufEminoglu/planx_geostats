# -*- coding: utf-8 -*-
"""PlanX GeoStats Lab spatial statistics suite for QGIS.

Registers the PlanX GeoStats Lab provider and its GeoStats Libraries helper.
Analysis algorithms appear under the Processing Toolbox.
"""
from __future__ import annotations

import os
from typing import Optional

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsApplication

from .dependencies import GeoStatsDependencyDialog
from .planx_geostats_provider import PlanXGeoStatsProvider


class PlanXGeoStatsPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.provider: Optional[PlanXGeoStatsProvider] = None
        self.dependencies_action: Optional[QAction] = None
        self.dependencies_dialog: Optional[GeoStatsDependencyDialog] = None

    def initGui(self) -> None:
        self.provider = PlanXGeoStatsProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

        icon = QIcon(os.path.join(self.plugin_dir, "icons", "icon.png"))
        self.dependencies_action = QAction(icon, "GeoStats Libraries", self.iface.mainWindow())
        self.dependencies_action.triggered.connect(self.open_dependencies)
        self.iface.addPluginToMenu("&PlanX", self.dependencies_action)
        self._warn_if_dependencies_missing()

    def unload(self) -> None:
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
        if self.dependencies_action is not None:
            self.iface.removePluginMenu("&PlanX", self.dependencies_action)
            self.dependencies_action = None
        self.dependencies_dialog = None

    def open_dependencies(self) -> None:
        if self.dependencies_dialog is None:
            self.dependencies_dialog = GeoStatsDependencyDialog(
                self.plugin_dir,
                self.iface.mainWindow(),
            )
        self.dependencies_dialog.refresh_status()
        self.dependencies_dialog.show()
        self.dependencies_dialog.raise_()
        self.dependencies_dialog.activateWindow()

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
            "Open PlanX > GeoStats Libraries to review and install them.",
        )
