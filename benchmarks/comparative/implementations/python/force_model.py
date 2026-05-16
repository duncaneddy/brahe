"""
Python (Brahe) function-level force-model acceleration benchmarks.

Each task repeatedly evaluates a single acceleration term at a fixed state.
This isolates the force-model code from the propagator/integrator.
"""

import numpy as np

import brahe

from benchmarks.comparative.implementations.python.base import (
    ensure_eop,
    time_iterations,
)
from benchmarks.comparative.results import TaskResult


def _epoch_and_state(params: dict) -> tuple[brahe.Epoch, np.ndarray]:
    epc = brahe.Epoch.from_jd(params["jd"], brahe.TimeSystem.UTC)
    state = np.array(params["state_eci"])
    return epc, state


def _result(task_name: str, iterations: int, times, results) -> TaskResult:
    return TaskResult(
        task_name=task_name,
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


def accel_point_mass_gravity(params: dict, iterations: int) -> TaskResult:
    """Evaluate central-body point-mass gravity acceleration."""
    ensure_eop()
    _, state = _epoch_and_state(params)
    r = state[:3]
    r_cb = np.zeros(3)
    n_samples = params["n_samples"]

    def run():
        a = None
        for _ in range(n_samples):
            a = brahe.accel_point_mass_gravity(r, r_cb, brahe.GM_EARTH)
        # Return acceleration as the result so it can be compared across libraries.
        return [a.tolist()]

    times, results = time_iterations(run, iterations)
    return _result("force_model.accel_point_mass_gravity", iterations, times, results)


def _accel_spherical_harmonics(
    task_name: str, params: dict, iterations: int
) -> TaskResult:
    ensure_eop()
    epc, state = _epoch_and_state(params)
    r = state[:3]
    n = params["degree"]
    m = params["order"]
    n_samples = params["n_samples"]

    gravity_model = brahe.GravityModel.from_model_type(
        brahe.GravityModelType.EGM2008_360
    )

    def run():
        a = None
        for _ in range(n_samples):
            rot = brahe.rotation_eci_to_ecef(epc)
            a = brahe.accel_gravity_spherical_harmonics(r, rot, gravity_model, n, m)
        return [a.tolist()]

    times, results = time_iterations(run, iterations)
    return _result(task_name, iterations, times, results)


def accel_spherical_harmonics_20(params: dict, iterations: int) -> TaskResult:
    return _accel_spherical_harmonics(
        "force_model.accel_spherical_harmonics_20", params, iterations
    )


def accel_spherical_harmonics_80(params: dict, iterations: int) -> TaskResult:
    return _accel_spherical_harmonics(
        "force_model.accel_spherical_harmonics_80", params, iterations
    )


def accel_third_body_sun(params: dict, iterations: int) -> TaskResult:
    """Evaluate Sun third-body acceleration using DE440s ephemeris."""
    ensure_eop()
    epc, state = _epoch_and_state(params)
    r = state[:3]
    n_samples = params["n_samples"]

    def run():
        a = None
        for _ in range(n_samples):
            a = brahe.accel_third_body_sun_de(epc, r, brahe.EphemerisSource.DE440s)
        return [a.tolist()]

    times, results = time_iterations(run, iterations)
    return _result("force_model.accel_third_body_sun", iterations, times, results)


def accel_third_body_moon(params: dict, iterations: int) -> TaskResult:
    """Evaluate Moon third-body acceleration using DE440s ephemeris."""
    ensure_eop()
    epc, state = _epoch_and_state(params)
    r = state[:3]
    n_samples = params["n_samples"]

    def run():
        a = None
        for _ in range(n_samples):
            a = brahe.accel_third_body_moon_de(epc, r, brahe.EphemerisSource.DE440s)
        return [a.tolist()]

    times, results = time_iterations(run, iterations)
    return _result("force_model.accel_third_body_moon", iterations, times, results)
