"""
Python (Brahe) time system conversion benchmarks.
"""

import brahe

from benchmarks.comparative.implementations.python.base import (
    ensure_eop,
    time_iterations,
)
from benchmarks.comparative.results import TaskResult


def _make_epochs(datetimes: list[dict]) -> list:
    """Create Epoch objects from datetime dicts."""
    epochs = []
    for dt in datetimes:
        epc = brahe.Epoch(
            dt["year"],
            dt["month"],
            dt["day"],
            dt["hour"],
            dt["minute"],
            dt["second"],
            dt["nanosecond"],
            time_system=brahe.TimeSystem.UTC,
        )
        epochs.append(epc)
    return epochs


def epoch_creation(params: dict, iterations: int) -> TaskResult:
    """Benchmark Epoch creation from datetime and JD extraction."""
    ensure_eop()
    datetimes = params["datetimes"]

    def run():
        results = []
        for dt in datetimes:
            epc = brahe.Epoch(
                dt["year"],
                dt["month"],
                dt["day"],
                dt["hour"],
                dt["minute"],
                dt["second"],
                dt["nanosecond"],
                time_system=brahe.TimeSystem.UTC,
            )
            results.append(epc.jd())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="time.epoch_creation",
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


def utc_to_tai(params: dict, iterations: int) -> TaskResult:
    """Benchmark UTC to TAI conversion."""
    ensure_eop()
    datetimes = params["datetimes"]
    epochs = _make_epochs(datetimes)

    def run():
        results = []
        for epc in epochs:
            jd_tai = epc.jd_as_time_system(brahe.TimeSystem.TAI)
            results.append(jd_tai)
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="time.utc_to_tai",
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


def utc_to_tt(params: dict, iterations: int) -> TaskResult:
    """Benchmark UTC to TT conversion."""
    ensure_eop()
    datetimes = params["datetimes"]
    epochs = _make_epochs(datetimes)

    def run():
        results = []
        for epc in epochs:
            jd_tt = epc.jd_as_time_system(brahe.TimeSystem.TT)
            results.append(jd_tt)
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="time.utc_to_tt",
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


def utc_to_gps(params: dict, iterations: int) -> TaskResult:
    """Benchmark UTC to GPS conversion."""
    ensure_eop()
    datetimes = params["datetimes"]
    epochs = _make_epochs(datetimes)

    def run():
        results = []
        for epc in epochs:
            jd_gps = epc.jd_as_time_system(brahe.TimeSystem.GPS)
            results.append(jd_gps)
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="time.utc_to_gps",
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


def utc_to_ut1(params: dict, iterations: int) -> TaskResult:
    """Benchmark UTC to UT1 conversion."""
    ensure_eop()
    datetimes = params["datetimes"]
    epochs = _make_epochs(datetimes)

    def run():
        results = []
        for epc in epochs:
            jd_ut1 = epc.jd_as_time_system(brahe.TimeSystem.UT1)
            results.append(jd_ut1)
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="time.utc_to_ut1",
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
