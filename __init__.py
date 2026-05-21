# -*- coding: utf-8 -*-
"""QGIS plugin entry point for PlanX GeoStats Lab."""
from __future__ import annotations

import logging

logger = logging.getLogger("PlanX GeoStats Lab")

# Check for spatial statistics dependencies
DEPENDENCIES_MISSING = False
MISSING_LIBS = []

for lib in ["libpysal", "esda", "spreg", "mgwr", "sklearn"]:
    try:
        if lib == "sklearn":
            import sklearn
        elif lib == "libpysal":
            import libpysal
        elif lib == "esda":
            import esda
        elif lib == "spreg":
            import spreg
        elif lib == "mgwr":
            import mgwr
    except ImportError:
        DEPENDENCIES_MISSING = True
        MISSING_LIBS.append(lib)

if DEPENDENCIES_MISSING:
    logger.warning("Missing dependencies for PlanX GeoStats Lab: %s", ", ".join(MISSING_LIBS))
else:
    logger.info("PlanX GeoStats Lab dependencies loaded successfully.")

from .main_plugin import PlanXGeoStatsPlugin


def classFactory(iface):
    """Factory function loaded by QGIS to instantiate the plugin."""
    return PlanXGeoStatsPlugin(iface)
