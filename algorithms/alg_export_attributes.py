# -*- coding: utf-8 -*-
"""Export Attributes and Coordinates to CSV/ASCII Processing Algorithm."""
from __future__ import annotations

import logging
import csv
from qgis.core import (
    NULL,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFileDestination,
)

from ..core.weights import geometry_centroid_point
from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class ExportAttributesAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELDS = "FIELDS"
    DELIMITER = "DELIMITER"
    INCLUDE_COORDS = "INCLUDE_COORDS"
    OUTPUT_FILE = "OUTPUT_FILE"

    def name(self) -> str:
        return "export_attributes_to_ascii"

    def displayName(self) -> str:
        return "Export Feature Attributes to CSV/ASCII"

    def group(self) -> str:
        return "01 | Data Preparation and Neighborhoods"

    def groupId(self) -> str:
        return "planx_prepare_neighbors"

    def icon(self):
        return algorithm_icon("export_attributes_to_ascii")

    def createInstance(self):
        return ExportAttributesAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Exports selected attributes and feature centroid coordinates (X, Y) "
            "to a formatted text file (CSV, Tab-delimited, Space-delimited, or Semicolon-delimited).\n\n"
            "This is helpful for preparing spatial datasets to be analyzed in external statistical "
            "software packages like R, SPSS, SAS, or Python."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELDS,
                "Fields to export (optional, blank exports all)",
                parentLayerParameterName=self.INPUT,
                allowMultiple=True,
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DELIMITER,
                "Delimiter",
                options=["Comma (,)", "Tab (\\t)", "Semicolon (;)", "Space ( )"],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.INCLUDE_COORDS,
                "Include geometry centroid coordinates (X, Y)",
                defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_FILE,
                "Output ASCII text file",
                fileFilter="Text/CSV files (*.csv *.txt *.tsv *.asc)",
                defaultValue=""
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        export_fields = self.parameterAsFields(parameters, self.FIELDS, context)
        delim_idx = self.parameterAsEnum(parameters, self.DELIMITER, context)
        include_coords = self.parameterAsBoolean(parameters, self.INCLUDE_COORDS, context)
        out_filepath = self.parameterAsFileOutput(parameters, self.OUTPUT_FILE, context)

        if not out_filepath:
            raise QgsProcessingException("Output file path must be specified.")

        # Determine delimiter character
        delimiters = [',', '\t', ';', ' ']
        delim = delimiters[delim_idx]

        # Resolve fields to export
        all_fields = [f.name() for f in source.fields()]
        if not export_fields:
            export_fields = all_fields

        # Get field indexes
        field_idxs = []
        for name in export_fields:
            idx = source.fields().lookupField(name)
            if idx < 0:
                raise QgsProcessingException(f"Field '{name}' not found in layer.")
            field_idxs.append(idx)

        # Build CSV header
        headers = []
        if include_coords:
            headers.extend(["X_COORD", "Y_COORD"])
        headers.extend(export_fields)

        feedback.pushInfo(f"Exporting attributes to file: {out_filepath}")

        total = source.featureCount() or 1
        with open(out_filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=delim)
            writer.writerow(headers)

            for idx, feature in enumerate(source.getFeatures()):
                if feedback.isCanceled():
                    break

                row = []

                # Add coordinates
                if include_coords:
                    geom = feature.geometry()
                    centroid = geometry_centroid_point(geom)
                    if centroid is not None:
                        row.extend([centroid.x(), centroid.y()])
                    else:
                        row.extend([None, None])

                # Add fields
                for f_idx in field_idxs:
                    val = feature.attribute(f_idx)
                    if val == NULL or val is None or str(val) == 'NULL':
                        row.append('')
                    else:
                        row.append(str(val))

                writer.writerow(row)
                feedback.setProgress(int(100 * (idx / total)))

        return {self.OUTPUT_FILE: out_filepath}
