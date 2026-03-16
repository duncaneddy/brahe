"""
Python (Brahe) coordinate conversion benchmarks.
"""

import numpy as np

import brahe

from benchmarks.comparative.implementations.python.base import (
    ensure_eop,
    time_iterations,
)
from benchmarks.comparative.results import TaskResult


def geodetic_to_ecef(params: dict, iterations: int) -> TaskResult:
    """Benchmark geodetic to ECEF conversion using brahe."""
    ensure_eop()
    points = params["points"]

    def run():
        results = []
        for lon, lat, alt in points:
            geod = np.array([lon, lat, alt])
            ecef = brahe.position_geodetic_to_ecef(geod, brahe.AngleFormat.DEGREES)
            results.append(ecef.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="coordinates.geodetic_to_ecef",
        language="python",
        library="brahe",
        iterations=iterations,
        times_seconds=times,
        results=results,
        metadata={
            "library": "brahe",
            "language": "python",
            "version": getattr(brahe, "__version__", "unknown"),
        },
    )


def ecef_to_geodetic(params: dict, iterations: int) -> TaskResult:
    """Benchmark ECEF to geodetic conversion using brahe."""
    ensure_eop()
    points = params["points"]

    def run():
        results = []
        for x, y, z in points:
            ecef = np.array([x, y, z])
            geod = brahe.position_ecef_to_geodetic(ecef, brahe.AngleFormat.DEGREES)
            results.append(geod.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="coordinates.ecef_to_geodetic",
        language="python",
        library="brahe",
        iterations=iterations,
        times_seconds=times,
        results=results,
        metadata={
            "library": "brahe",
            "language": "python",
            "version": getattr(brahe, "__version__", "unknown"),
        },
    )


def geocentric_to_ecef(params: dict, iterations: int) -> TaskResult:
    """Benchmark geocentric to ECEF conversion using brahe."""
    ensure_eop()
    points = params["points"]

    def run():
        results = []
        for lon, lat, radius in points:
            geoc = np.array([lon, lat, radius])
            ecef = brahe.position_geocentric_to_ecef(geoc, brahe.AngleFormat.DEGREES)
            results.append(ecef.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="coordinates.geocentric_to_ecef",
        language="python",
        library="brahe",
        iterations=iterations,
        times_seconds=times,
        results=results,
        metadata={
            "library": "brahe",
            "language": "python",
            "version": getattr(brahe, "__version__", "unknown"),
        },
    )


def ecef_to_geocentric(params: dict, iterations: int) -> TaskResult:
    """Benchmark ECEF to geocentric conversion using brahe."""
    ensure_eop()
    points = params["points"]

    def run():
        results = []
        for x, y, z in points:
            ecef = np.array([x, y, z])
            geoc = brahe.position_ecef_to_geocentric(ecef, brahe.AngleFormat.DEGREES)
            results.append(geoc.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="coordinates.ecef_to_geocentric",
        language="python",
        library="brahe",
        iterations=iterations,
        times_seconds=times,
        results=results,
        metadata={
            "library": "brahe",
            "language": "python",
            "version": getattr(brahe, "__version__", "unknown"),
        },
    )


def ecef_to_azel(params: dict, iterations: int) -> TaskResult:
    """Benchmark ECEF to azimuth/elevation/range conversion using brahe."""
    ensure_eop()
    pairs = params["pairs"]

    def run():
        results = []
        for pair in pairs:
            station = np.array(pair["station_ecef"])
            satellite = np.array(pair["satellite_ecef"])
            enz = brahe.relative_position_ecef_to_enz(
                station,
                satellite,
                brahe.EllipsoidalConversionType.GEODETIC,
            )
            azel = brahe.position_enz_to_azel(enz, brahe.AngleFormat.DEGREES)
            results.append(azel.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="coordinates.ecef_to_azel",
        language="python",
        library="brahe",
        iterations=iterations,
        times_seconds=times,
        results=results,
        metadata={
            "library": "brahe",
            "language": "python",
            "version": getattr(brahe, "__version__", "unknown"),
        },
    )
