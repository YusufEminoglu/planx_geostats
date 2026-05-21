# -*- coding: utf-8 -*-
"""GeoStats dependency helper UI."""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import List, Optional, Tuple

from qgis.PyQt.QtCore import QProcess, Qt
from qgis.PyQt.QtGui import QTextCursor
from qgis.PyQt.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PIP_PACKAGES = ["libpysal", "esda", "spreg", "mgwr", "scikit-learn"]
MODULES = {
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "libpysal": "libpysal",
    "esda": "esda",
    "spreg": "spreg",
    "mgwr": "mgwr",
}


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


class GeoStatsDependencyDialog(QDialog):
    """User-confirmed installer for optional GeoStats Python libraries."""

    GUIDE_TEXT = (
        "PlanX GeoStats Lab can run its core Processing tools with QGIS and NumPy, "
        "but several advanced spatial statistics workflows become more complete when "
        "the PySAL family and scikit-learn are available inside the same Python "
        "environment that QGIS is using. This panel checks that environment directly; "
        "it does not inspect your system Python, Anaconda, PyCharm interpreter, or any "
        "other Python installation that QGIS cannot import from.\n\n"
        "The listed libraries support different parts of the GeoStats workflow. NumPy "
        "is the numerical foundation used throughout the algorithms. libpysal provides "
        "spatial weights and neighborhood structures used by autocorrelation methods. "
        "esda provides established exploratory spatial data analysis statistics such "
        "as Moran and Getis-Ord variants. spreg supports spatial-regression oriented "
        "diagnostics and the Spatial Lag and Spatial Error regression tools. mgwr supports Multiscale "
        "Geographically Weighted Regression with variable-specific bandwidths. scikit-learn supports nearest-neighbor "
        "search, clustering, and standardized multivariate analysis routines.\n\n"
        "Use 'QGIS Python pip' when QGIS already has a working Python executable and "
        "internet access. On Windows OSGeo4W installations, 'OSGeo Shell' can be useful "
        "because it starts the OSGeo environment first and then runs pip from there. "
        "Both modes show the exact command before anything runs. Nothing is installed "
        "silently: press 'Install / Update' only after reviewing the command and after "
        "you are comfortable allowing pip to modify the active QGIS Python environment.\n\n"
        "After a successful installation, restart QGIS completely. Python modules are "
        "loaded into the running process, so a restart is the cleanest way for the "
        "Processing provider, reports, and symbology helpers to see newly installed "
        "packages. If installation fails, keep the log visible; proxy settings, locked "
        "site-packages folders, missing build wheels, or using the wrong Python "
        "environment are the most common causes."
    )

    def __init__(self, plugin_dir: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.plugin_dir = plugin_dir
        self.process: Optional[QProcess] = None
        self.setWindowTitle("PlanX GeoStats Libraries")
        self.resize(860, 720)

        self.host_label = QLabel(sys.executable)
        self.host_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.python_label = QLabel(resolve_qgis_python_executable() or "Not found")
        self.python_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.guide_box = QTextEdit()
        self.guide_box.setReadOnly(True)
        self.guide_box.setPlainText(self.GUIDE_TEXT)
        self.guide_box.setMinimumHeight(210)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(150)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("QGIS Python pip", "qgis_python")
        if sys.platform == "win32":
            self.mode_combo.addItem("OSGeo Shell", "osgeo_shell")
        self.mode_combo.currentIndexChanged.connect(self.update_command_preview)

        self.command_preview = QTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMinimumHeight(82)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_status)
        self.copy_button = QPushButton("Copy Command")
        self.copy_button.clicked.connect(self.copy_command)
        self.install_button = QPushButton("Install Missing / Update Libraries")
        self.install_button.clicked.connect(self.confirm_and_install)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("QGIS host application", self.host_label)
        form.addRow("Python used for pip", self.python_label)
        form.addRow("Install mode", self.mode_combo)

        guide_group = QGroupBox("Guide")
        guide_layout = QVBoxLayout(guide_group)
        guide_layout.addWidget(self.guide_box)

        status_group = QGroupBox("GeoStats dependency status")
        status_layout = QVBoxLayout(status_group)
        status_layout.addWidget(self.summary_label)
        status_layout.addWidget(self.status_box)

        command_group = QGroupBox("Command preview")
        command_layout = QVBoxLayout(command_group)
        command_layout.addWidget(self.command_preview)

        actions = QHBoxLayout()
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.copy_button)
        actions.addStretch(1)
        actions.addWidget(self.install_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(guide_group)
        layout.addWidget(status_group)
        layout.addWidget(command_group)
        layout.addLayout(actions)
        layout.addWidget(QLabel("Installation log"))
        layout.addWidget(self.log_box, 1)
        layout.addWidget(buttons)

        self.refresh_status()

    def reject(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            answer = QMessageBox.question(
                self,
                "PlanX GeoStats Libraries",
                "An installation process is still running. Stop it and close?",
            )
            if answer != QMessageBox.Yes:
                return
            self.process.kill()
        super().reject()

    def refresh_status(self) -> None:
        self.python_label.setText(resolve_qgis_python_executable() or "Not found")
        lines = []
        missing = []
        for package, module in MODULES.items():
            ok = importlib.util.find_spec(module) is not None
            marker = "OK" if ok else "MISSING"
            if not ok:
                missing.append(package)
            role = self._package_role(package)
            lines.append(f"{marker:7} {package} ({module})\n        {role}")
        if missing:
            self.summary_label.setText(
                "Missing libraries: "
                + ", ".join(missing)
                + ". Review the command preview before installing."
            )
            self.install_button.setText("Install Missing / Update Libraries")
        else:
            self.summary_label.setText(
                "All checked GeoStats libraries are available in the active QGIS Python environment. "
                "You can still run an update if you intentionally want pip to upgrade them."
            )
            self.install_button.setText("Update Libraries")
        self.status_box.setPlainText("\n".join(lines))
        self.update_command_preview()

    def update_command_preview(self) -> None:
        try:
            program, args = self._build_command()
            self.command_preview.setPlainText(self._format_command(program, args))
            self.copy_button.setEnabled(True)
        except RuntimeError as exc:
            self.command_preview.setPlainText(str(exc))
            self.copy_button.setEnabled(False)

    def copy_command(self) -> None:
        command = self.command_preview.toPlainText().strip()
        if not command or command.startswith("OSGeo Shell was selected"):
            QMessageBox.information(
                self,
                "PlanX GeoStats Libraries",
                "There is no runnable command to copy for the selected install mode.",
            )
            return
        QApplication.clipboard().setText(command)
        self.log_box.append("Command copied to the clipboard.")

    def confirm_and_install(self) -> None:
        try:
            program, args = self._build_command()
        except RuntimeError as exc:
            QMessageBox.warning(self, "PlanX GeoStats Libraries", str(exc))
            return

        command_text = self._format_command(program, args)
        answer = QMessageBox.question(
            self,
            "PlanX GeoStats Libraries",
            "Review the command below before continuing. It will modify the Python "
            "environment used by QGIS, not a separate project interpreter.\n\n"
            f"{command_text}\n\n"
            "QGIS should be restarted after installation.",
        )
        if answer != QMessageBox.Yes:
            self.log_box.append("Installation canceled before running any command.")
            return

        self.install_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.copy_button.setEnabled(False)
        self.log_box.clear()
        self.log_box.append(f"$ {command_text}")

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._append_process_output)
        self.process.finished.connect(self._installation_finished)
        self.process.start(program, args)

        if not self.process.waitForStarted(3000):
            self.install_button.setEnabled(True)
            self.refresh_button.setEnabled(True)
            self.copy_button.setEnabled(True)
            QMessageBox.warning(
                self,
                "PlanX GeoStats Libraries",
                "Could not start installation process.",
            )

    def _append_process_output(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.log_box.moveCursor(QTextCursor.End)
            self.log_box.insertPlainText(data)
            self.log_box.moveCursor(QTextCursor.End)

    def _installation_finished(self, exit_code: int, _exit_status) -> None:
        self.install_button.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.copy_button.setEnabled(True)
        self.log_box.append("")
        if exit_code == 0:
            self.log_box.append("Dependencies installed. Restart QGIS before running GeoStats tools.")
            QMessageBox.information(
                self,
                "PlanX GeoStats Libraries",
                "Dependencies installed successfully. Please restart QGIS.",
            )
        else:
            self.log_box.append(f"Installation failed with exit code {exit_code}.")
            QMessageBox.warning(
                self,
                "PlanX GeoStats Libraries",
                f"Installation failed with exit code {exit_code}. See the log for details.",
            )
        self.refresh_status()

    def _build_command(self) -> Tuple[str, List[str]]:
        packages = self._read_requirements()
        mode = self.mode_combo.currentData()
        if mode == "qgis_python":
            return build_qgis_python_pip_command(packages)
        if mode == "osgeo_shell":
            return build_osgeo_shell_pip_command(packages)
        raise RuntimeError("Unknown installation mode.")

    def _read_requirements(self) -> List[str]:
        path = os.path.join(self.plugin_dir, "requirements_geostats.txt")
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

    def _format_command(self, program: str, args: List[str]) -> str:
        return format_command(program, args)

    def _quote(self, value: str) -> str:
        return quote_command_part(value)

    def _package_role(self, package: str) -> str:
        roles = {
            "numpy": "Numerical arrays, matrix operations, and core statistical calculations.",
            "scikit-learn": "Nearest-neighbor search, clustering support, and standardized multivariate workflows.",
            "libpysal": "Spatial weights, neighborhood graphs, and PySAL-compatible spatial structures.",
            "esda": "Exploratory spatial data analysis statistics such as Moran and Getis-Ord routines.",
            "spreg": "Spatial regression utilities for model-oriented workflows, including Spatial Lag and Spatial Error regression.",
            "mgwr": "Multiscale geographically weighted regression with variable-specific bandwidths.",
        }
        return roles.get(package, "Optional GeoStats support package.")
