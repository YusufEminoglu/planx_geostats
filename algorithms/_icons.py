# -*- coding: utf-8 -*-
"""Shared icon lookup for PlanX GeoStats Lab Processing algorithms."""
from __future__ import annotations

import os

from qgis.PyQt.QtGui import QIcon


def algorithm_icon(algorithm_id: str) -> QIcon:
    icon_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "icons",
        "algorithms",
        f"{algorithm_id}.png",
    )
    if os.path.exists(icon_path):
        return QIcon(icon_path)
    fallback = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons", "icon.png")
    return QIcon(fallback) if os.path.exists(fallback) else QIcon()
