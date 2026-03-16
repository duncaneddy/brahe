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
