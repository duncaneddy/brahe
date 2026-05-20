"""Basilisk orbital-element conversion benchmarks.

Basilisk's orbitalMotion.ClassicElements uses true anomaly (f) in radians;
brahe's KOE vector uses mean anomaly (M) in degrees. Conversion between the
two conventions happens outside time_iterations so the timed call measures
only the native orbitalMotion.elem2rv / rv2elem operation.
"""

import math

import numpy as np

from Basilisk.utilities import orbitalMotion

from benchmarks.comparative.implementations.basilisk.base import (
    build_task_result,
    time_iterations,
)

# Match brahe/Orekit's mu constant (m^3/s^2). The Java baseline uses
# Constants.EIGEN5C_EARTH_MU; brahe uses brahe.GM_EARTH. We use the same
# numeric value (3.986004418e14) so the timed call sees the same gravity
# parameter all three baselines do.
MU_EARTH = 3.986004418e14


def _oe_brahe_to_basilisk(oe_deg) -> "orbitalMotion.ClassicElements":
    """Convert [a, e, i, raan, argp, M] deg -> Basilisk ClassicElements (rad, true anomaly)."""
    a, e, i_deg, raan_deg, argp_deg, M_deg = oe_deg
    M = math.radians(M_deg)
    # Solve Kepler's equation for E, then E -> f
    E = orbitalMotion.M2E(M, e)
    f = orbitalMotion.E2f(E, e)
    oe = orbitalMotion.ClassicElements()
    oe.a = a
    oe.e = e
    oe.i = math.radians(i_deg)
    oe.Omega = math.radians(raan_deg)
    oe.omega = math.radians(argp_deg)
    oe.f = f
    return oe


def _oe_basilisk_to_brahe(oe) -> list[float]:
    """Convert Basilisk ClassicElements (rad, true anomaly) -> [a, e, i, raan, argp, M] deg."""
    E = orbitalMotion.f2E(oe.f, oe.e)
    M = orbitalMotion.E2M(E, oe.e)
    return [
        float(oe.a),
        float(oe.e),
        math.degrees(float(oe.i)),
        math.degrees(float(oe.Omega)),
        math.degrees(float(oe.omega)),
        math.degrees(M),
    ]


def keplerian_to_cartesian(params: dict, iterations: int):
    """Convert Keplerian elements [a, e, i, raan, argp, M] (deg) to Cartesian state [r, v]."""
    # Pre-convert input to Basilisk's convention OUTSIDE the timed region.
    bsk_elements = [_oe_brahe_to_basilisk(oe) for oe in params["elements"]]

    def run():
        results = []
        for oe in bsk_elements:
            r, v = orbitalMotion.elem2rv(MU_EARTH, oe)
            results.append([float(r[0]), float(r[1]), float(r[2]),
                            float(v[0]), float(v[1]), float(v[2])])
        return results

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "orbits.keplerian_to_cartesian", iterations, times, results
    )


def cartesian_to_keplerian(params: dict, iterations: int):
    """Convert Cartesian state [r, v] to Keplerian elements [a, e, i, raan, argp, M] (deg)."""
    states = [np.array(s, dtype=float) for s in params["states"]]

    def run():
        # Timed region: pure Basilisk native call, output in Basilisk's
        # (rad, true anomaly) convention.
        return [
            orbitalMotion.rv2elem(MU_EARTH, s[:3], s[3:6])
            for s in states
        ]

    times, native_results = time_iterations(run, iterations)
    # Post-convert OUTSIDE the timed region so timing reflects native call only.
    results = [_oe_basilisk_to_brahe(oe) for oe in native_results]
    return build_task_result(
        "orbits.cartesian_to_keplerian", iterations, times, results
    )
