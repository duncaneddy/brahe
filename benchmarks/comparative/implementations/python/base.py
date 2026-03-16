"""
Base utilities for Python benchmark implementations.
"""

import time

import brahe


def ensure_eop():
    """Initialize EOP if not already done."""
    if not brahe.get_global_eop_initialization():
        eop = brahe.StaticEOPProvider.from_zero()
        brahe.set_global_eop_provider(eop)


def time_iterations(func, iterations: int) -> tuple[list[float], list]:
    """Time a function over multiple iterations.

    Returns (times_seconds, results_from_first_iteration).
    """
    times = []
    first_results = None

    for i in range(iterations):
        start = time.perf_counter()
        results = func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if i == 0:
            first_results = results

    return times, first_results
