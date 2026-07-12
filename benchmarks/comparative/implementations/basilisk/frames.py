"""Basilisk frame-transformation benchmarks via pyswice (SPICE Toolkit).

Implementation note: Basilisk does not expose an ECI->ECEF state transform
as a standalone function — Basilisk users typically go through the bundled
pyswice (Spice Toolkit) interface. We use pyswice.sxform_c between J2000
and ITRF93 (high-precision Earth body-fixed frame backed by NAIF's
earth_latest_high_prec.bpc binary PCK kernel). ITRF93 closely matches
Orekit's ITRF (IERS 2010 conventions); expect cm-to-m-scale agreement.

Kernels (LSK + text PCK + binary high-precision Earth PCK) are loaded once
at module import time so kernel loading does not contaminate the
per-iteration timing. The high-precision PCK is downloaded by
`just _bench-compare-build-basilisk` and lives in ~/.bsk-data/.

pyswice uses SWIG-generated C-style out-parameter wrappers. The idiom for
scalar output is new_doubleArray(1) / doubleArray_getitem, and for the 6x6
state-transform matrix the output is a flat 36-element double array in
row-major order.
"""

import os
from pathlib import Path

import numpy as np

from Basilisk.topLevelModules import pyswice
from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path

from benchmarks.comparative.implementations.basilisk.base import (
    build_task_result,
    jd_to_spice_et_string,
    time_iterations,
)

# Load required SPICE kernels exactly once. furnsh is idempotent for the same
# kernel file but redundant calls are still cheap.
_KERNELS_LOADED = False

_HIGH_PREC_PCK = Path(
    os.environ.get(
        "BSK_HIGH_PREC_PCK",
        str(Path.home() / ".cache" / "bsk-data" / "earth_latest_high_prec.bpc"),
    )
)


def _ensure_kernels() -> None:
    global _KERNELS_LOADED
    if _KERNELS_LOADED:
        return
    pyswice.furnsh_c(str(get_path(DataFile.EphemerisData.naif0012)))  # LSK
    pyswice.furnsh_c(str(get_path(DataFile.EphemerisData.pck00010)))  # text PCK
    if not _HIGH_PREC_PCK.exists():
        raise RuntimeError(
            f"High-precision Earth PCK not found at {_HIGH_PREC_PCK}. "
            "Run `just _bench-compare-build-basilisk` to download it."
        )
    pyswice.furnsh_c(
        str(_HIGH_PREC_PCK)
    )  # binary PCK with ITRF93 high-precision rotation
    _KERNELS_LOADED = True


def _jd_to_et(jd: float) -> float:
    """Julian Date UTC -> SPICE ET (seconds past J2000 TDB)."""
    et_arr = pyswice.new_doubleArray(1)
    pyswice.str2et_c(jd_to_spice_et_string(jd), et_arr)
    return pyswice.doubleArray_getitem(et_arr, 0)


def _sxform(from_frame: str, to_frame: str, et: float) -> np.ndarray:
    """Return the 6x6 SPICE state transformation matrix as a (6,6) numpy array."""
    xform_arr = pyswice.new_doubleArray(36)
    pyswice.sxform_c(from_frame, to_frame, et, xform_arr)
    flat = [pyswice.doubleArray_getitem(xform_arr, i) for i in range(36)]
    return np.array(flat, dtype=float).reshape(6, 6)


def state_eci_to_ecef(params: dict, iterations: int):
    """Transform 6D states from J2000 (ECI) to ITRF93 (ECEF) at given epochs."""
    _ensure_kernels()
    cases = params["cases"]
    # Pre-compute ET and prepare state arrays OUTSIDE the timed region.
    # pyswice.sxform_c expects km and km/s; we convert metres/mps -> km/kps and back.
    et_state_pairs = []
    for case in cases:
        et = _jd_to_et(case["jd"])
        state = np.array(case["state"], dtype=float)  # [x, y, z, vx, vy, vz] m/mps
        et_state_pairs.append((et, state))

    def run():
        results = []
        for et, state in et_state_pairs:
            # 6x6 state transformation matrix J2000 -> ITRF93
            sxform = _sxform("J2000", "ITRF93", et)
            s_km = np.concatenate([state[:3] / 1000.0, state[3:6] / 1000.0])
            ecef_km = sxform @ s_km
            ecef = np.concatenate([ecef_km[:3] * 1000.0, ecef_km[3:6] * 1000.0])
            results.append(ecef.tolist())
        return results

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "frames.state_eci_to_ecef",
        iterations,
        times,
        results,
        extra_metadata={"implementation": "pyswice", "earth_fixed_frame": "ITRF93"},
    )


def state_ecef_to_eci(params: dict, iterations: int):
    """Transform 6D states from ITRF93 (ECEF) to J2000 (ECI) at given epochs."""
    _ensure_kernels()
    cases = params["cases"]
    et_state_pairs = []
    for case in cases:
        et = _jd_to_et(case["jd"])
        state = np.array(case["state"], dtype=float)
        et_state_pairs.append((et, state))

    def run():
        results = []
        for et, state in et_state_pairs:
            sxform = _sxform("ITRF93", "J2000", et)
            s_km = np.concatenate([state[:3] / 1000.0, state[3:6] / 1000.0])
            eci_km = sxform @ s_km
            eci = np.concatenate([eci_km[:3] * 1000.0, eci_km[3:6] * 1000.0])
            results.append(eci.tolist())
        return results

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "frames.state_ecef_to_eci",
        iterations,
        times,
        results,
        extra_metadata={"implementation": "pyswice", "earth_fixed_frame": "ITRF93"},
    )
