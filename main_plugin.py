# -*- coding: utf-8 -*-
"""PlanX GeoStats Lab spatial statistics suite for QGIS.

Registers the PlanX GeoStats Lab provider. All user-facing tools appear under
the Processing Toolbox, including setup and dependency diagnostics.
"""
from __future__ import annotations

from typing import Optional

from qgis.core import QgsApplication

from .planx_geostats_provider import PlanXGeoStatsProvider


class PlanXGeoStatsPlugin:
    DIAGNOSTICS_PATH = "PlanX GeoStats Lab > 00 | Setup and Diagnostics"

    def __init__(self, iface):
        self.iface = iface
        self.provider: Optional[PlanXGeoStatsProvider] = None

    def initGui(self) -> None:
        self.provider = PlanXGeoStatsProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)
        self._warn_if_dependencies_missing()

    def unload(self) -> None:
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
