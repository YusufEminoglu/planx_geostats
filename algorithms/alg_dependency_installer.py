# -*- coding: utf-8 -*-
"""Algorithm to install spatial statistics dependencies."""
from __future__ import annotations

import sys
import subprocess
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterBoolean,
    QgsProcessingOutputString
)


class DependencyInstallerAlgorithm(QgsProcessingAlgorithm):
    FORCE_REINSTALL = "FORCE_REINSTALL"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "install_dependencies"

    def displayName(self) -> str:
        return "Install/Update SpatialStats Python Dependencies"

    def group(self) -> str:
        return "Utilities"

    def groupId(self) -> str:
        return "utilities"

    def createInstance(self):
        return DependencyInstallerAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Installs PySAL (libpysal, esda, spreg) and scikit-learn "
            "into QGIS's active Python environment using pip.\n\n"
            "This algorithm runs in a background thread. After installation, "
            "please restart QGIS to enable all advanced algorithms."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FORCE_REINSTALL,
                "Force reinstall/upgrade packages",
                defaultValue=False
            )
        )
        self.addOutput(QgsProcessingOutputString(self.OUTPUT, "Installation Log"))

    def processAlgorithm(self, parameters, context, feedback):
        force = self.parameterAsBool(parameters, self.FORCE_REINSTALL, context)
        libs = ["libpysal", "esda", "spreg", "scikit-learn"]

        cmd = [sys.executable, "-m", "pip", "install"]
        if force:
            cmd.append("--upgrade")
            cmd.append("--force-reinstall")
        cmd.extend(libs)

        feedback.pushInfo(f"Running installation command: {' '.join(cmd)}")

        startupinfo = None
        if sys.platform == 'win32':
            # Hide the cmd window on Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            startupinfo=startupinfo
        )

        log = []
        while True:
            if feedback.isCanceled():
                process.terminate()
                feedback.pushInfo("Installation canceled by user.")
                break

            line = process.stdout.readline()
            if not line:
                break

            line_str = line.strip()
            if line_str:
                feedback.pushInfo(line_str)
                log.append(line_str)

        process.wait()

        if process.returncode == 0:
            feedback.pushInfo(
                "Dependencies installed successfully!\n"
                "Please restart QGIS to load the spatial statistics tools."
            )
            # Update the global flag dynamically
            try:
                import planx_geostats
                planx_geostats.DEPENDENCIES_MISSING = False
            except Exception:
                pass
        else:
            feedback.reportError(f"Installation failed with exit code {process.returncode}")

        return {self.OUTPUT: "\n".join(log)}
