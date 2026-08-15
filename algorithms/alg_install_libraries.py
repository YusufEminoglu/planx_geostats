# -*- coding: utf-8 -*-
"""User-approved GeoStats library installer Processing Algorithm."""
from __future__ import annotations

import os
# Subprocess is limited to this explicit, user-approved setup algorithm.
import subprocess  # nosec B404
import sys
import time
from typing import List, Tuple

from ._mixins import HelpUrlMixin
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
)

from ..dependencies import (
    PIP_PACKAGES,
    build_osgeo_shell_pip_command,
    build_qgis_python_pip_command,
    find_osgeo_shell,
    format_command,
    resolve_qgis_python_executable,
)


from ._icons import algorithm_icon


class InstallGeoStatsLibrariesAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INSTALL_MODE = "INSTALL_MODE"
    CONFIRM = "CONFIRM"
    COMMAND = "COMMAND"

    MODES = ["QGIS Python pip", "OSGeo Shell"]

    def name(self) -> str:
        return "install_geostats_libraries"

    def displayName(self) -> str:
        return "Install / Update GeoStats Libraries"

    def group(self) -> str:
        return "00 | Setup and Diagnostics"

    def groupId(self) -> str:
        return "planx_setup_diagnostics"

    def icon(self):
        return algorithm_icon("install_geostats_libraries")

    def createInstance(self):
        return InstallGeoStatsLibrariesAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Installs or updates the optional Python libraries used by PlanX "
            "GeoStats Lab (numba, libpysal, esda, spreg, mgwr, scikit-learn, xgboost, "
            "lightgbm, shap, catboost, interpret, mapie, dice-ml, tabpfn) into the "
            "active QGIS Python environment. The "
            "package list is read from requirements_geostats.txt, or falls back to "
            "the same fourteen packages if that file is missing. tabpfn additionally "
            "needs an internet connection the first time it actually runs (not during "
            "this pip install), to download its pretrained model weights.\n\n"
            "This tool is intentionally not silent. With the confirmation checkbox "
            "left unchecked it prints the exact command, the QGIS host executable, "
            "the resolved Python executable, and the detected OSGeo4W shell, then "
            "stops before pip runs. Enable the checkbox only after reviewing that "
            "preview.\n\n"
            "Two installation modes are offered. QGIS Python pip runs the resolved "
            "python.exe with '-m pip install --upgrade <packages>'; pick this when "
            "that executable was resolved automatically. OSGeo Shell instead calls "
            "OSGeo4W.bat first to set up the environment before invoking pip; use it "
            "on Windows when QGIS was launched from the qgis-bin.exe application "
            "executable and no bare Python interpreter is available.\n\n"
            "Typical failure causes: no Python interpreter could be resolved (switch "
            "to OSGeo Shell mode), a permission error writing into a protected "
            "site-packages directory (run as administrator or use a user-site "
            "install), or a package version with no wheel for the QGIS Python "
            "version.\n\n"
            "After a successful install, restart QGIS before running MGWR, Spatial "
            "Autoregression, or Spatial Error Regression - Python caches imported "
            "modules for the life of the process, so a package installed mid-session "
            "is not visible to it."
        )

    def initAlgorithm(self, config=None):
        default_mode = self._default_mode_index()
        self.addParameter(
            QgsProcessingParameterEnum(
                self.INSTALL_MODE,
                "Installation mode",
                options=self.MODES,
                defaultValue=default_mode,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CONFIRM,
                "I have reviewed this tool and approve running pip in the QGIS Python environment",
                defaultValue=False,
            )
        )
        self.addOutput(QgsProcessingOutputString(self.COMMAND, "Executed command"))

    def processAlgorithm(self, parameters, context, feedback):
        mode = self.parameterAsEnum(parameters, self.INSTALL_MODE, context)
        packages = self._read_requirements()
        program, args = self._build_command(mode, packages)
        command_text = format_command(program, args)

        feedback.pushInfo("PlanX GeoStats Lab dependency installation")
        feedback.pushInfo(f"Command: {command_text}")
        feedback.pushInfo(f"QGIS host application executable: {sys.executable}")
        feedback.pushInfo(f"Python executable selected for pip: {resolve_qgis_python_executable() or 'not found'}")
        feedback.pushInfo(f"Detected OSGeo Shell: {find_osgeo_shell() or 'not found'}")

        approved = self.parameterAsBoolean(parameters, self.CONFIRM, context)
        if not approved:
            raise QgsProcessingException(
                "Preview only: installation was not started. Review the command above, "
                "enable the confirmation checkbox, and run the tool again to execute pip."
            )

        feedback.pushInfo("The process may take several minutes. Keep QGIS open until it finishes.")

        exit_code = self._run_process(program, args, feedback)
        if exit_code != 0:
            raise QgsProcessingException(
                f"GeoStats library installation failed with exit code {exit_code}. "
                "Review the Processing log above for the pip error."
            )

        feedback.pushInfo("GeoStats libraries installed or updated successfully.")
        feedback.pushInfo("Restart QGIS before running advanced GeoStats tools.")
        return {self.COMMAND: command_text}

    def _default_mode_index(self) -> int:
        if resolve_qgis_python_executable():
            return 0
        if sys.platform.startswith("win") and find_osgeo_shell():
            return 1
        return 0

    def _build_command(self, mode: int, packages: List[str]) -> Tuple[str, List[str]]:
        if mode == 0:
            return build_qgis_python_pip_command(packages)
        if mode == 1:
            return build_osgeo_shell_pip_command(packages)
        raise QgsProcessingException("Unknown installation mode.")

    def _read_requirements(self) -> List[str]:
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(plugin_dir, "requirements_geostats.txt")
        if not os.path.exists(path):
            return list(PIP_PACKAGES)
        packages = []
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                packages.append(line)
        return packages or list(PIP_PACKAGES)

    def _run_process(self, program: str, args: List[str], feedback) -> int:
        # Command arguments are built from plugin constants after explicit user approval.
        process = subprocess.Popen(  # nosec B603
            [program] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is None:
            raise QgsProcessingException("Could not capture the installation process output.")
        while True:
            if feedback.isCanceled():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise QgsProcessingException("Installation was canceled by the user.")

            line = process.stdout.readline()
            if line:
                feedback.pushInfo(line.rstrip())
                continue

            if process.poll() is not None:
                break
            time.sleep(0.1)

        for remaining in process.stdout:
            if remaining:
                feedback.pushInfo(remaining.rstrip())
        return process.returncode
