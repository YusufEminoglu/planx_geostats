# -*- coding: utf-8 -*-
"""Dependency command helpers for PlanX GeoStats Lab setup algorithms."""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple


PIP_PACKAGES = ["libpysal", "esda", "spreg", "mgwr", "scikit-learn"]
MODULES = {
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "libpysal": "libpysal",
    "esda": "esda",
    "spreg": "spreg",
    "mgwr": "mgwr",
}


def optional_dependency_error(tool_name: str, packages: List[str], import_error: Exception) -> str:
    """Return consistent Processing guidance for optional GeoStats library failures."""
    package_list = ", ".join(packages)
    preview = ""
    try:
        program, args = build_qgis_python_pip_command(list(packages))
        preview = f" Preview command: {format_command(program, args)}."
    except Exception:
        preview = ""
    return (
        f"{tool_name} requires optional Python package(s): {package_list}. "
        "Run PlanX GeoStats Lab > 00 | Setup and Diagnostics > GeoStats Library Status "
        "to inspect the active QGIS Python environment, or run Install / Update GeoStats "
        "Libraries from the same toolbox group with explicit approval. "
        "Restart QGIS after installing packages because already-loaded Processing providers "
        "may not see newly installed modules."
        f"{preview} Import error: {import_error}"
    )


def resolve_qgis_python_executable() -> Optional[str]:
    """Return a Python executable that belongs to the running QGIS install."""
    executable = sys.executable
    name = os.path.basename(executable).lower()
    if name.startswith("python"):
        return executable

    candidates = []
    executable_dir = os.path.dirname(executable)
    if executable_dir:
        candidates.extend([
            os.path.join(executable_dir, "python.exe"),
            os.path.join(executable_dir, "python3.exe"),
        ])

    osgeo_root = os.environ.get("OSGEO4W_ROOT")
    if osgeo_root:
        candidates.extend([
            os.path.join(osgeo_root, "bin", "python.exe"),
            os.path.join(osgeo_root, "bin", "python3.exe"),
        ])

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def find_osgeo_shell() -> Optional[str]:
    candidates = []
    osgeo_root = os.environ.get("OSGEO4W_ROOT")
    if osgeo_root:
        candidates.append(os.path.join(osgeo_root, "OSGeo4W.bat"))

    executable_dir = os.path.dirname(sys.executable)
    if executable_dir:
        install_root = os.path.dirname(executable_dir)
        candidates.append(os.path.join(install_root, "OSGeo4W.bat"))

    candidates.extend([
        r"C:\OSGeo4W\OSGeo4W.bat",
        r"C:\OSGeo4W64\OSGeo4W.bat",
    ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def quote_command_part(value: str) -> str:
    if not value:
        return '""'
    if any(ch.isspace() for ch in value) or any(ch in value for ch in '()&'):
        return f'"{value}"'
    return value


def format_command(program: str, args: List[str]) -> str:
    return " ".join(quote_command_part(part) for part in [program] + list(args))


def build_qgis_python_pip_command(packages: List[str]) -> Tuple[str, List[str]]:
    python_executable = resolve_qgis_python_executable()
    if python_executable is None:
        raise RuntimeError(
            "QGIS is running from an application executable, and a Python executable "
            "could not be found beside it. Use OSGeo Shell mode or run the status "
            "report to inspect the detected paths."
        )
    return python_executable, ["-m", "pip", "install", "--upgrade"] + list(packages)


def build_osgeo_shell_pip_command(packages: List[str]) -> Tuple[str, List[str]]:
    bat = find_osgeo_shell()
    if bat is None:
        raise RuntimeError(
            "OSGeo Shell was selected, but OSGeo4W.bat could not be found. "
            "Use QGIS Python pip or set OSGEO4W_ROOT."
        )
    pip_command = " ".join(["python", "-m", "pip", "install", "--upgrade"] + list(packages))
    command = f'call "{bat}" && {pip_command}'
    return "cmd.exe", ["/c", command]
