"""
Python (Brahe) frame transformation benchmarks.
"""

import numpy as np

import brahe

from benchmarks.comparative.implementations.python.base import (
    ensure_eop,
    time_iterations,
)
from benchmarks.comparative.results import TaskResult


def state_eci_to_ecef(params: dict, iterations: int) -> TaskResult:
    """Benchmark ECI to ECEF state transformation using brahe."""
    ensure_eop()
    cases = params["cases"]

    # Pre-build Epoch objects
    epoch_state_pairs = []
    for case in cases:
        epc = brahe.Epoch.from_jd(case["jd"], brahe.TimeSystem.UTC)
        state = np.array(case["state"])
        epoch_state_pairs.append((epc, state))

    def run():
        results = []
        for epc, state in epoch_state_pairs:
            ecef = brahe.state_eci_to_ecef(epc, state)
            results.append(ecef.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="frames.state_eci_to_ecef",
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


def state_ecef_to_eci(params: dict, iterations: int) -> TaskResult:
    """Benchmark ECEF to ECI state transformation using brahe."""
    ensure_eop()
    cases = params["cases"]

    # Pre-build Epoch objects
    epoch_state_pairs = []
    for case in cases:
        epc = brahe.Epoch.from_jd(case["jd"], brahe.TimeSystem.UTC)
        state = np.array(case["state"])
        epoch_state_pairs.append((epc, state))

    def run():
        results = []
        for epc, state in epoch_state_pairs:
            eci = brahe.state_ecef_to_eci(epc, state)
            results.append(eci.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="frames.state_ecef_to_eci",
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
