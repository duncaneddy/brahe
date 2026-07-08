"""
Python (Brahe) function-level force-model acceleration benchmarks.

Each task either:

- Perf path (single IC): evaluates a single acceleration term at one fixed
  state, repeating ``n_samples`` times per iteration to amortize call
  overhead. Returns one acceleration vector.
- Accuracy path (``cases`` IC sweep): iterates over N cases, evaluating
  the acceleration once per case. Returns N acceleration vectors so the
  accuracy harness can compare them per-sample and build a distribution.

This isolates the force-model code from the propagator/integrator.
"""

from typing import Callable

import numpy as np

import brahe

from benchmarks.comparative.implementations.python.base import (
    ensure_eop,
    time_iterations,
)
from benchmarks.comparative.results import TaskResult


def _state_eci_array(state_list) -> np.ndarray:
    return np.array(state_list)


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


def _run_perf_or_sweep(
    params: dict,
    eval_for_case: Callable[[brahe.Epoch, np.ndarray], list[float]],
) -> Callable[[], list]:
    """Return the zero-arg ``run`` closure shared by every force-model task.

    Dispatches on whether ``params`` is the single-IC perf shape or the
    multi-IC accuracy shape (``cases`` present). Each ``eval_for_case``
    must return the acceleration as a flat 3-list.
    """
    cases = params.get("cases")
    if cases is None:
        epc = brahe.Epoch.from_jd(params["jd"], brahe.TimeSystem.UTC)
        state = _state_eci_array(params["state_eci"])
        n_samples = params["n_samples"]

        def run():
            a = None
            for _ in range(n_samples):
                a = eval_for_case(epc, state)
            return [a]

        return run

    def run_sweep():
        out = []
        for case in cases:
            epc = brahe.Epoch.from_jd(case["jd"], brahe.TimeSystem.UTC)
            state = _state_eci_array(case["state_eci"])
            out.append(eval_for_case(epc, state))
        return out

    return run_sweep


def accel_point_mass_gravity(params: dict, iterations: int) -> TaskResult:
    """Evaluate central-body point-mass gravity acceleration."""
    ensure_eop()
    r_cb = np.zeros(3)

    def eval_case(_epc: brahe.Epoch, state: np.ndarray) -> list[float]:
        a = brahe.accel_point_mass_gravity(state[:3], r_cb, brahe.GM_EARTH)
        return a.tolist()

    times, results = time_iterations(_run_perf_or_sweep(params, eval_case), iterations)
    return _result("force_model.accel_point_mass_gravity", iterations, times, results)


def _accel_spherical_harmonics(
    task_name: str, params: dict, iterations: int
) -> TaskResult:
    ensure_eop()
    n = params["degree"]
    m = params["order"]

    gravity_model = brahe.GravityModel.from_model_type(
        brahe.GravityModelType.EGM2008_360
    )

    def eval_case(epc: brahe.Epoch, state: np.ndarray) -> list[float]:
        rot = brahe.rotation_eci_to_ecef(epc)
        a = brahe.accel_gravity_spherical_harmonics(state[:3], rot, gravity_model, n, m)
        return a.tolist()

    times, results = time_iterations(_run_perf_or_sweep(params, eval_case), iterations)
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

    def eval_case(epc: brahe.Epoch, state: np.ndarray) -> list[float]:
        a = brahe.accel_third_body_sun_de(epc, state[:3], brahe.EphemerisSource.DE440s)
        return a.tolist()

    times, results = time_iterations(_run_perf_or_sweep(params, eval_case), iterations)
    return _result("force_model.accel_third_body_sun", iterations, times, results)


def accel_third_body_moon(params: dict, iterations: int) -> TaskResult:
    """Evaluate Moon third-body acceleration using DE440s ephemeris."""
    ensure_eop()

    def eval_case(epc: brahe.Epoch, state: np.ndarray) -> list[float]:
        a = brahe.accel_third_body_moon_de(epc, state[:3], brahe.EphemerisSource.DE440s)
        return a.tolist()

    times, results = time_iterations(_run_perf_or_sweep(params, eval_case), iterations)
    return _result("force_model.accel_third_body_moon", iterations, times, results)
