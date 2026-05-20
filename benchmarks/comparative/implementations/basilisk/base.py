"""Shared helpers for the Basilisk benchmark implementations.

Includes:
  - time_iterations: identical signature to the brahe-py helper.
  - j2000_to_gcrf: wraps brahe.state_eme2000_to_gcrf for Basilisk's N (J2000)
    inertial output.
  - build_task_result: factory for TaskResult with consistent metadata.
  - jd_to_utc_epoch: brahe Epoch from a Julian Date (UTC).
  - jd_to_spice_et_string: format a JD UTC instant as a SPICE-acceptable
    calendar string (e.g. "2024 JAN 1 00:00:00.000 UTC") for Basilisk's
    createSpiceInterface(time=...).
  - ensure_eop: same EOP-from-Orekit-finals pattern as implementations/python/base.py
    so brahe's frame transform has IERS data loaded.
"""

import os
import time
from pathlib import Path

import numpy as np

import brahe

from benchmarks.comparative.results import TaskResult


def time_iterations(func, iterations: int):
    """Time a function over multiple iterations.

    Returns (times_seconds, results_from_first_iteration).
    """
    times: list[float] = []
    first_results = None
    for i in range(iterations):
        start = time.perf_counter()
        results = func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if i == 0:
            first_results = results
    return times, first_results


def _find_orekit_eop_file() -> str | None:
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


def ensure_eop() -> None:
    """Initialize brahe EOP using OreKit's IERS data if available, else fallback.

    Identical behavior to implementations/python/base.py:ensure_eop so the
    J2000->GCRF transform applied to Basilisk output uses the same IERS data
    that the brahe-py and Orekit baselines use.
    """
    if not brahe.get_global_eop_initialization():
        eop_path = _find_orekit_eop_file()
        if eop_path:
            provider = brahe.FileEOPProvider.from_file(eop_path, True, "Hold")
            brahe.set_global_eop_provider(provider)
        else:
            brahe.initialize_eop()


def jd_to_utc_epoch(jd: float) -> "brahe.Epoch":
    """Construct a brahe Epoch (UTC) from a Julian Date."""
    return brahe.Epoch.from_jd(jd, brahe.TimeSystem.UTC)


def jd_to_spice_et_string(jd: float) -> str:
    """Format a JD (UTC) as a SPICE-acceptable calendar UTC string.

    Basilisk's gravFactory.createSpiceInterface(time=...) accepts a calendar
    string parsed by SPICE; we render an unambiguous ISO-style form.
    """
    epc = jd_to_utc_epoch(jd)
    # brahe.Epoch.isostring_with_decimals(3) returns "YYYY-MM-DDTHH:MM:SS.fffZ" -> reformat for SPICE
    iso = epc.isostring_with_decimals(3)  # e.g. "2024-01-01T00:00:00.000Z"
    date, rest = iso.split("T")
    t = rest.rstrip("Z")
    return f"{date} {t} UTC"


def j2000_to_gcrf(r_m, v_mps) -> list[float]:
    """Transform a 6-vector state from J2000 (EME2000) inertial to GCRF.

    Basilisk's N inertial frame is J2000-equatorial. brahe's
    state_eme2000_to_gcrf applies the IAU 2006 frame bias from EME2000 to GCRF
    (time-independent, ~sub-meter at LEO). EOP initialization is not strictly
    required for this transform but we ensure it for consistency with the
    rest of the runner.
    """
    ensure_eop()
    state = np.array([*r_m, *v_mps], dtype=float)
    gcrf = brahe.state_eme2000_to_gcrf(state)
    return list(map(float, gcrf))


def build_task_result(
    task_name: str,
    iterations: int,
    times_seconds: list,
    results: list,
    extra_metadata: dict | None = None,
) -> TaskResult:
    """Build a TaskResult with standard Basilisk metadata."""
    import Basilisk

    metadata = {
        "library": "basilisk",
        "language": "basilisk",
        "version": getattr(Basilisk, "__version__", "unknown"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return TaskResult(
        task_name=task_name,
        language="basilisk",
        library="basilisk",
        iterations=iterations,
        times_seconds=times_seconds,
        results=results,
        metadata=metadata,
    )
