# -*- coding: utf-8 -*-
"""GeoStats library status Processing Algorithm."""
from __future__ import annotations

import html
import importlib.util
import os
import sys
import tempfile

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingOutputHtml,
    QgsProcessingParameterFileDestination,
)

from ..dependencies import MODULES, PIP_PACKAGES


class GeoStatsLibraryStatusAlgorithm(QgsProcessingAlgorithm):
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "geostats_library_status"

    def displayName(self) -> str:
        return "GeoStats Library Status"

    def group(self) -> str:
        return "00 | Setup and Diagnostics"

    def groupId(self) -> str:
        return "planx_setup_diagnostics"

    def createInstance(self):
        return GeoStatsLibraryStatusAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Creates an HTML diagnostic report for the optional Python libraries used "
            "by PlanX GeoStats Lab. The report checks the active QGIS Python "
            "environment, explains which packages are available, and prints the exact "
            "pip command that can be reviewed or copied.\n\n"
            "This tool does not install anything. Use it when the menu helper is not "
            "visible, when documenting a QGIS profile, or when sending a dependency "
            "status report before troubleshooting."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "GeoStats library status report"))

    def processAlgorithm(self, parameters, context, feedback):
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_geostats_library_status.html")

        statuses = []
        for package, module in MODULES.items():
            available = importlib.util.find_spec(module) is not None
            statuses.append({
                "package": package,
                "module": module,
                "available": available,
                "role": self._package_role(package),
            })

        missing = [item["package"] for item in statuses if not item["available"]]
        command = self._install_command()
        feedback.pushInfo(f"QGIS Python executable: {sys.executable}")
        if missing:
            feedback.pushInfo("Missing GeoStats libraries: " + ", ".join(missing))
        else:
            feedback.pushInfo("All checked GeoStats libraries are available.")

        self._write_html(html_path, statuses, missing, command)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _install_command(self) -> str:
        parts = [sys.executable, "-m", "pip", "install", "--upgrade"] + list(PIP_PACKAGES)
        return " ".join(self._quote(part) for part in parts)

    def _quote(self, value: str) -> str:
        if not value:
            return '""'
        if any(ch.isspace() for ch in value) or any(ch in value for ch in '()&'):
            return f'"{value}"'
        return value

    def _package_role(self, package: str) -> str:
        roles = {
            "numpy": "Numerical arrays, matrix operations, and core statistical calculations.",
            "scikit-learn": "Nearest-neighbor search, clustering support, and standardized multivariate workflows.",
            "libpysal": "Spatial weights, neighborhood graphs, and PySAL-compatible spatial structures.",
            "esda": "Exploratory spatial data analysis statistics such as Moran and Getis-Ord routines.",
            "spreg": "Spatial regression utilities and diagnostics for model-oriented workflows.",
        }
        return roles.get(package, "Optional GeoStats support package.")

    def _write_html(self, path, statuses, missing, command):
        rows = []
        for item in statuses:
            state = "Available" if item["available"] else "Missing"
            state_class = "ok" if item["available"] else "missing"
            rows.append(
                "<tr>"
                f"<td>{html.escape(item['package'])}</td>"
                f"<td>{html.escape(item['module'])}</td>"
                f"<td class=\"{state_class}\">{state}</td>"
                f"<td>{html.escape(item['role'])}</td>"
                "</tr>"
            )

        if missing:
            summary = (
                "Missing optional libraries: "
                + html.escape(", ".join(missing))
                + ". Use PlanX GeoStats Lab > GeoStats Libraries for guided installation."
            )
        else:
            summary = "All checked optional GeoStats libraries are available in the active QGIS Python environment."

        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Library Status</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.65rem; }}
.subtitle {{ color: #64748b; margin: 0 0 22px; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 20px 0; }}
.command {{ background: #111827; color: #f9fafb; padding: 14px 16px; border-radius: 6px; font-family: Consolas, monospace; overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px; text-align: left; vertical-align: top; font-size: .9rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
.ok {{ color: #2f855a; font-weight: 700; }}
.missing {{ color: #c2410c; font-weight: 700; }}
</style>
</head>
<body>
<div class="container">
<h1>GeoStats Library Status</h1>
<p class="subtitle">Active QGIS Python executable: <strong>{html.escape(sys.executable)}</strong></p>
<div class="summary">{summary}</div>
<h2>Recommended install/update command</h2>
<p>Review this command before running it. It targets the Python environment reported above.</p>
<div class="command">{html.escape(command)}</div>
<h2>Package status</h2>
<table>
<thead><tr><th>Package</th><th>Import module</th><th>Status</th><th>GeoStats role</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
