"""
Base utilities for Python benchmark implementations.
"""

import os
import time
from pathlib import Path

import brahe


def _find_orekit_eop_file() -> str | None:
    """Find the OreKit EOP finals2000A.all file."""
    orekit_data = os.environ.get(
        "OREKIT_DATA", str(Path.home() / ".orekit" / "orekit-data")
    )
    eop_path = (
        Path(orekit_data)
        / "Earth-Orientation-Parameters"
        / "IAU-2000"
        / "finals2000A.all"
    )
    if eop_path.exists():
        return str(eop_path)
    return None


def ensure_eop():
    """Initialize EOP using OreKit's real IERS data if available, else fallback."""
    if not brahe.get_global_eop_initialization():
        eop_path = _find_orekit_eop_file()
        if eop_path:
            provider = brahe.FileEOPProvider.from_file(eop_path, True, "Hold")
            brahe.set_global_eop_provider(provider)
        else:
            # Fallback: download from IERS
            brahe.initialize_eop()


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
