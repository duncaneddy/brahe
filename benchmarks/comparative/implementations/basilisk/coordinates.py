"""Basilisk geodetic <-> ECEF coordinate conversions via pyswice.

Basilisk does not ship a standalone geodetic conversion; users go through
pyswice. We use georec_c (geodetic -> rectangular) and recgeo_c (rectangular
-> geodetic) with the WGS84 ellipsoid parameters that the existing brahe
benchmark uses (equatorial radius 6378137.0 m, flattening 1/298.257223563).

pyswice uses SWIG-generated C-style out-parameter wrappers. The idiom for
array output is new_doubleArray(N) / doubleArray_getitem, and for passing an
input array new_doubleArray(N) / doubleArray_setitem.
"""

import math

import numpy as np

from Basilisk.topLevelModules import pyswice

from benchmarks.comparative.implementations.basilisk.base import (
    build_task_result,
    time_iterations,
)

# WGS84 — matches brahe's defaults and the Java baseline.
_RE_KM = 6378.137
_FLAT = 1.0 / 298.257223563


def _georec(lon: float, lat: float, alt_km: float) -> tuple[float, float, float]:
    """Wrap pyswice.georec_c (geodetic -> rectangular). Returns (x, y, z) in km."""
    out = pyswice.new_doubleArray(3)
    pyswice.georec_c(lon, lat, alt_km, _RE_KM, _FLAT, out)
    return (
        pyswice.doubleArray_getitem(out, 0),
        pyswice.doubleArray_getitem(out, 1),
        pyswice.doubleArray_getitem(out, 2),
    )


def _recgeo(x_km: float, y_km: float, z_km: float) -> tuple[float, float, float]:
    """Wrap pyswice.recgeo_c (rectangular -> geodetic). Returns (lon_rad, lat_rad, alt_km)."""
    rectan = pyswice.new_doubleArray(3)
    pyswice.doubleArray_setitem(rectan, 0, x_km)
    pyswice.doubleArray_setitem(rectan, 1, y_km)
    pyswice.doubleArray_setitem(rectan, 2, z_km)
    lon_out = pyswice.new_doubleArray(1)
    lat_out = pyswice.new_doubleArray(1)
    alt_out = pyswice.new_doubleArray(1)
    pyswice.recgeo_c(rectan, _RE_KM, _FLAT, lon_out, lat_out, alt_out)
    return (
        pyswice.doubleArray_getitem(lon_out, 0),
        pyswice.doubleArray_getitem(lat_out, 0),
        pyswice.doubleArray_getitem(alt_out, 0),
    )


def geodetic_to_ecef(params: dict, iterations: int):
    """Convert (lon_deg, lat_deg, alt_m) -> ECEF (x, y, z) meters."""
    # Pre-convert lon/lat to radians and alt to km OUTSIDE the timed region.
    points_rad_km = [
        (math.radians(lon), math.radians(lat), alt / 1000.0)
        for lon, lat, alt in params["points"]
    ]

    def run():
        results = []
        for lon, lat, alt_km in points_rad_km:
            x_km, y_km, z_km = _georec(lon, lat, alt_km)
            results.append([x_km * 1000.0, y_km * 1000.0, z_km * 1000.0])
        return results

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "coordinates.geodetic_to_ecef",
        iterations,
        times,
        results,
        extra_metadata={"implementation": "pyswice", "ellipsoid": "WGS84"},
    )


def ecef_to_geodetic(params: dict, iterations: int):
    """Convert ECEF (x, y, z) meters -> (lon_deg, lat_deg, alt_m)."""
    points_km = [
        (x / 1000.0, y / 1000.0, z / 1000.0) for x, y, z in params["points"]
    ]

    def run():
        # Timed region: native pyswice call only.
        return [_recgeo(x, y, z) for x, y, z in points_km]

    times, native_results = time_iterations(run, iterations)
    # recgeo_c returns (lon_rad, lat_rad, alt_km). Convert OUTSIDE the timed region.
    results = [
        [math.degrees(lon), math.degrees(lat), alt_km * 1000.0]
        for lon, lat, alt_km in native_results
    ]
    return build_task_result(
        "coordinates.ecef_to_geodetic",
        iterations,
        times,
        results,
        extra_metadata={"implementation": "pyswice", "ellipsoid": "WGS84"},
    )
