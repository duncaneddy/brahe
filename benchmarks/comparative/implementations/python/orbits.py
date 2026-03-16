"""
Python (Brahe) orbital element conversion benchmarks.
"""

import numpy as np

import brahe

from benchmarks.comparative.implementations.python.base import (
    ensure_eop,
    time_iterations,
)
from benchmarks.comparative.results import TaskResult


def keplerian_to_cartesian(params: dict, iterations: int) -> TaskResult:
    """Benchmark Keplerian to Cartesian conversion using brahe."""
    ensure_eop()
    elements = params["elements"]

    def run():
        results = []
        for oe in elements:
            x_oe = np.array(oe)
            cart = brahe.state_koe_to_eci(x_oe, brahe.AngleFormat.DEGREES)
            results.append(cart.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="orbits.keplerian_to_cartesian",
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


def cartesian_to_keplerian(params: dict, iterations: int) -> TaskResult:
    """Benchmark Cartesian to Keplerian conversion using brahe."""
    ensure_eop()
    states = params["states"]

    def run():
        results = []
        for state in states:
            x_cart = np.array(state)
            oe = brahe.state_eci_to_koe(x_cart, brahe.AngleFormat.DEGREES)
            results.append(oe.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="orbits.cartesian_to_keplerian",
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
