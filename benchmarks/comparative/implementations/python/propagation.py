"""
Python (Brahe) propagation benchmarks.
"""

import numpy as np

import brahe

from benchmarks.comparative.implementations.python.base import (
    ensure_eop,
    time_iterations,
)
from benchmarks.comparative.results import TaskResult


def keplerian_single(params: dict, iterations: int) -> TaskResult:
    """Benchmark Keplerian propagation of multiple orbits to single future epochs."""
    ensure_eop()
    cases = params["cases"]

    def run():
        results = []
        for case in cases:
            epc = brahe.Epoch.from_jd(case["jd"], brahe.TimeSystem.UTC)
            oe = np.array(case["elements"])
            dt = case["dt"]
            target = epc + dt

            # Convert Keplerian to Cartesian so state() returns Cartesian
            cart = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)
            prop = brahe.KeplerianPropagator.from_eci(epc, cart, 60.0)
            state = prop.state(target)
            results.append(state.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="propagation.keplerian_single",
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


def keplerian_trajectory(params: dict, iterations: int) -> TaskResult:
    """Benchmark Keplerian propagation over trajectory steps."""
    ensure_eop()
    jd = params["jd"]
    oe = np.array(params["elements"])
    step_size = params["step_size"]
    n_steps = params["n_steps"]

    epc = brahe.Epoch.from_jd(jd, brahe.TimeSystem.UTC)

    # Convert Keplerian to Cartesian so current_state() returns Cartesian
    cart = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)

    def run():
        prop = brahe.KeplerianPropagator.from_eci(epc, cart, step_size)
        results = []
        for step_idx in range(n_steps):
            target = epc + (step_idx + 1) * step_size
            state = prop.state(target)
            results.append(state.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="propagation.keplerian_trajectory",
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


def sgp4_single(params: dict, iterations: int) -> TaskResult:
    """Benchmark SGP4 propagation to 50 future epochs."""
    ensure_eop()
    line1 = params["line1"]
    line2 = params["line2"]
    offsets = params["time_offsets_seconds"]

    prop = brahe.SGPPropagator.from_tle(line1, line2, 60.0)
    base_epoch = prop.epoch

    def run():
        results = []
        for dt in offsets:
            target = base_epoch + dt
            # Use .state() to get TEME output directly (matches Java's TEME frame)
            state = prop.state(target)
            results.append(state.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="propagation.sgp4_single",
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


def sgp4_trajectory(params: dict, iterations: int) -> TaskResult:
    """Benchmark SGP4 propagation over 1 day at 60s steps."""
    ensure_eop()
    line1 = params["line1"]
    line2 = params["line2"]
    step_size = params["step_size"]
    n_steps = params["n_steps"]

    def run():
        prop = brahe.SGPPropagator.from_tle(line1, line2, step_size)
        base_epoch = prop.epoch
        results = []
        for step_idx in range(n_steps):
            target = base_epoch + (step_idx + 1) * step_size
            # Use .state() to get TEME output directly (matches Java's TEME frame)
            state = prop.state(target)
            results.append(state.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="propagation.sgp4_trajectory",
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


def numerical_twobody(params: dict, iterations: int) -> TaskResult:
    """Benchmark numerical two-body propagation over 1 orbital period."""
    ensure_eop()
    jd = params["jd"]
    oe = np.array(params["elements"])
    step_size = params["step_size"]
    n_steps = params["n_steps"]

    epc = brahe.Epoch.from_jd(jd, brahe.TimeSystem.UTC)

    # Convert Keplerian elements to Cartesian state for numerical propagator
    cart = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)

    prop_config = brahe.NumericalPropagationConfig.default()
    force_config = brahe.ForceModelConfig.two_body()

    def run():
        prop = brahe.NumericalOrbitPropagator(
            epc,
            cart,
            prop_config,
            force_config,
        )
        prop.set_trajectory_mode(brahe.TrajectoryMode.DISABLED)
        results = []
        for step_idx in range(n_steps):
            target = epc + (step_idx + 1) * step_size
            prop.propagate_to(target)
            state = prop.current_state()
            results.append(state.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="propagation.numerical_twobody",
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
