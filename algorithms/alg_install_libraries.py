# -*- coding: utf-8 -*-
"""User-approved GeoStats library installer Processing Algorithm."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import List, Tuple

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
    format_command,
)


class InstallGeoStatsLibrariesAlgorithm(QgsProcessingAlgorithm):
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

    def createInstance(self):
        return InstallGeoStatsLibrariesAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Installs or updates the optional Python libraries used by PlanX "
            "GeoStats Lab: libpysal, esda, spreg, and scikit-learn. The tool reads "
            "the package list from requirements_geostats.txt, builds a pip command "
            "for the Python environment used by QGIS, and streams the installation "
            "log into the Processing feedback panel.\n\n"
            "This tool is intentionally not silent. It will not run until the "
            "confirmation checkbox is enabled. The command is printed before "
            "execution so you can review exactly which executable and packages will "
            "be used. After a successful installation, restart QGIS before running "
            "advanced GeoStats workflows because Python imports are cached in the "
            "current QGIS session.\n\n"
            "Use QGIS Python pip when a python.exe belonging to the active QGIS "
            "installation is detected. Use OSGeo Shell on Windows when QGIS was "
            "launched through an OSGeo4W application executable and direct Python "
            "resolution is not available."
        )

    def initAlgorithm(self, config=None):
        default_mode = 1 if sys.platform.startswith("win") else 0
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
        approved = self.parameterAsBoolean(parameters, self.CONFIRM, context)
        if not approved:
            raise QgsProcessingException(
                "Installation was not started. Enable the confirmation checkbox after "
                "reviewing the tool description and run it again."
            )

        mode = self.parameterAsEnum(parameters, self.INSTALL_MODE, context)
        packages = self._read_requirements()
        program, args = self._build_command(mode, packages)
        command_text = format_command(program, args)

        feedback.pushInfo("PlanX GeoStats Lab dependency installation")
        feedback.pushInfo(f"Command: {command_text}")
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
        process = subprocess.Popen(
            [program] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
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
