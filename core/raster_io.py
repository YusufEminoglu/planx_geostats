# -*- coding: utf-8 -*-
"""GeoTIFF output for the interpolation tools.

These are the plugin's first raster outputs.  They are written through
``QgsRasterFileWriter`` rather than the GDAL Python bindings, so no ``osgeo``
import is needed and the CRS travels as a QGIS object rather than a WKT
string.

Four behaviours are encoded here because each one fails *quietly* if assumed,
and each was checked by running it on both supported runtimes (QGIS 3.44 LTR
and QGIS 4.2):

1. ``provider.setEditable(False)`` is what finalises the file.  Without it the
   GeoTIFF is left as a 360-byte stub and only becomes complete when the
   provider object happens to be garbage collected.  A surface that is
   truncated now and readable a minute later is worse than one that fails
   immediately, so a write is not finished until that call has succeeded.
2. Row 0 of the array is the NORTH row, matching the raster convention.  The
   cell centres built by :func:`grid_centres` are ordered to match.
3. A destination that cannot be created (a missing directory, a read-only
   volume) is reported by ``createOneBandRaster`` returning ``None`` or an
   invalid provider.  It does not raise, so both are checked.
4. Block writes take ``xOffset`` / ``yOffset``, not ``offsetX`` / ``offsetY``.
"""
from __future__ import annotations

import logging
import math
import os
import re

import numpy as np

from qgis.core import (
    Qgis,
    QgsRasterBlock,
    QgsRasterFileWriter,
)

logger = logging.getLogger("PlanX GeoStats Lab")

#: Sentinel written for cells that could not be predicted.  Chosen to sit far
#: outside any plausible interpolated value so it is obvious on the map.
DEFAULT_NODATA = -9999.0

#: Rows written per block.  Bounds the peak size of the in-memory block without
#: making the write loop long enough to matter.
BLOCK_ROWS = 256

#: Ceiling on the number of cells in one interpolated surface.  A 4-million
#: cell grid is 16 MB per band as Float32, but the cost that matters is the
#: kriging prediction, which is linear in cells: measured at roughly 300 cells
#: per second for 2000 input points, so this ceiling is already several minutes
#: of work.  It exists to turn an accidental ``CELL_SIZE`` of 0.001 into a
#: message rather than an overnight run.
MAX_CELLS = 4_000_000

#: Longest side of the automatic grid, in cells.  A square data extent
#: therefore defaults to roughly 200 x 200 = 40 000 cells.
AUTO_GRID_TARGET = 200

_EXT_RE = re.compile(r"\*\.([A-Za-z0-9]+)")

#: Extensions whose driver must not be decided by scan order.  GeoTIFF and COG
#: both advertise ``*.tif``; a prediction surface is an ordinary local file, so
#: GeoTIFF is the right one.
_PREFERRED_DRIVERS = {"tif": "GTiff", "tiff": "GTiff"}


def supported_driver_extensions():
    """Map a lowercase file extension to a GDAL driver short name.

    Built from what this QGIS build actually supports rather than a hard-coded
    list, so it stays correct if a runtime ships a different GDAL.
    """
    mapping = dict(_PREFERRED_DRIVERS)
    try:
        formats = QgsRasterFileWriter.supportedFiltersAndFormats()
    except (AttributeError, TypeError):  # pragma: no cover - binding drift
        return mapping
    for details in formats:
        driver = details.driverName
        if not isinstance(driver, str):
            driver = driver()
        pattern = details.filterString
        if not isinstance(pattern, str):
            pattern = pattern()
        for ext in _EXT_RE.findall(pattern):
            mapping.setdefault(ext.lower(), driver)
    return mapping


def driver_for_path(path):
    """Pick a GDAL driver for ``path`` from its extension.

    Returns ``(driver, note)``.  ``note`` is a warning when the extension is not
    one this build can write and GeoTIFF is being used instead, so the caller
    can tell the user rather than leaving them with a file whose name
    misdescribes its contents.  A path with no extension gets GeoTIFF silently,
    since there is no claim to contradict.
    """
    ext = os.path.splitext(str(path))[1].lstrip(".").lower()
    if not ext:
        return "GTiff", None
    mapping = supported_driver_extensions()
    driver = mapping.get(ext)
    if driver:
        return driver, None
    return "GTiff", (
        f"'{path}' has the extension '.{ext}', which is not a raster format "
        "this QGIS build can write. The file is being written as GeoTIFF "
        "whatever its name says; give it a .tif extension to avoid confusion."
    )


def auto_cell_size(extent):
    """A cell size giving roughly ``AUTO_GRID_TARGET`` cells on the long side."""
    span = max(float(extent.width()), float(extent.height()))
    if not math.isfinite(span) or span <= 0:
        return 0.0
    return span / float(AUTO_GRID_TARGET)


def grid_shape(extent, cell_size):
    """Number of ``(columns, rows)`` cells covering ``extent``.

    The count is rounded to the nearest whole cell and the grid is then stretched
    to cover the extent exactly, so the raster's footprint equals the data's
    rather than overhanging it.  The effective resolution can therefore differ
    slightly from the requested cell size, which is why the algorithms report
    the resolution they actually used.

    Raises ``ValueError`` for a degenerate extent or a grid past ``MAX_CELLS``,
    naming the count so the message is actionable.
    """
    width = float(extent.width())
    height = float(extent.height())
    if not (math.isfinite(width) and math.isfinite(height)):
        raise ValueError(
            "The data extent is not finite, so no output grid can be built. "
            "Check the input layer's coordinates."
        )
    if width <= 0 or height <= 0:
        raise ValueError(
            "The input features span zero "
            f"{'width' if width <= 0 else 'height'} in map units, so there is no "
            "area to interpolate over. At least one feature must differ in both "
            "coordinates."
        )
    size = float(cell_size)
    if not math.isfinite(size) or size <= 0:
        raise ValueError(f"Cell size must be a positive number; got {cell_size}.")

    # A single row or column is a line, not a surface: kriging it would work but
    # the result is not something a user can read as a map.
    cols = max(2, int(round(width / size)))
    rows = max(2, int(round(height / size)))
    cells = cols * rows
    if cells > MAX_CELLS:
        needed = math.sqrt(width * height / MAX_CELLS)
        raise ValueError(
            f"A cell size of {size:.6g} map units would need {cells:,} cells, "
            f"above this tool's ceiling of {MAX_CELLS:,}. Use a cell size of at "
            f"least {needed:.6g} map units, or narrow the analysis extent."
        )
    return cols, rows


def grid_centres(extent, cols, rows):
    """Cell-centre coordinates as ``(xs, ys)`` 1-D arrays.

    ``ys`` runs from north to south, and the arrays are meant to be combined
    with ``numpy.meshgrid(xs, ys)`` so that row 0 of the result is the northern
    row -- matching how the raster stores its rows.
    """
    res_x = float(extent.width()) / cols
    res_y = float(extent.height()) / rows
    xs = float(extent.xMinimum()) + (np.arange(cols) + 0.5) * res_x
    ys = float(extent.yMaximum()) - (np.arange(rows) + 0.5) * res_y
    return xs, ys


def write_float32_geotiff(path, array, extent, crs, nodata=DEFAULT_NODATA,
                          driver=None, feedback=None):
    """Write a 2-D array as a single-band Float32 raster.

    Row 0 of ``array`` becomes the northern row.  Non-finite cells are replaced
    by ``nodata``, because a NaN in a GeoTIFF is read back inconsistently by
    different drivers, whereas a declared no-data value is unambiguous.

    Returns the path written.  Raises ``ValueError`` for a bad array or grid and
    ``RuntimeError`` when the file itself cannot be created or finalised.
    """
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D surface; got shape {arr.shape}.")
    arr = np.asarray(arr, dtype=np.float32)
    rows, cols = arr.shape
    if rows < 1 or cols < 1:
        raise ValueError(f"Refusing to write an empty raster ({rows} x {cols}).")

    if not np.all(np.isfinite(arr)):
        replaced = int((~np.isfinite(arr)).sum())
        logger.debug("Replacing %d non-finite cell(s) with no-data", replaced)
        arr = np.where(np.isfinite(arr), arr, np.float32(nodata))

    note = None
    if driver is None:
        driver, note = driver_for_path(path)
    if note and feedback is not None:
        feedback.pushWarning(note)

    writer = QgsRasterFileWriter(str(path))
    writer.setOutputFormat(driver)
    provider = writer.createOneBandRaster(
        Qgis.Float32, cols, rows, extent, crs
    )
    if provider is None or not provider.isValid():
        raise RuntimeError(
            f"The raster '{path}' could not be created. Check that the folder "
            "exists and is writable, and that the file is not open in another "
            "program."
        )

    if not provider.isEditable():
        provider.setEditable(True)
    provider.setNoDataValue(1, float(nodata))

    for start in range(0, rows, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, rows)
        block = QgsRasterBlock(Qgis.Float32, cols, stop - start)
        block.setData(arr[start:stop, :].tobytes())
        if not provider.writeBlock(block, 1, xOffset=0, yOffset=start):
            raise RuntimeError(
                f"Writing rows {start}-{stop - 1} of '{path}' failed. The "
                "output may be incomplete; delete it before retrying."
            )

    # This is the call that flushes the file. Without it the raster stays a
    # truncated stub until the provider is garbage collected.
    if not provider.setEditable(False):
        raise RuntimeError(
            f"The raster '{path}' could not be finalised. Delete it and retry; "
            "if it persists, the output folder may be full or read-only."
        )
    return str(path)
