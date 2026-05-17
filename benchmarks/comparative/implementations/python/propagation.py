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

            prop = brahe.KeplerianPropagator.from_keplerian(
                epc, oe, brahe.AngleFormat.DEGREES, 60.0
            )
            state = prop.state_eci(target)
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

    def run():
        prop = brahe.KeplerianPropagator.from_keplerian(
            epc, oe, brahe.AngleFormat.DEGREES, step_size
        )
        results = []
        for step_idx in range(n_steps):
            target = epc + (step_idx + 1) * step_size
            state = prop.state_eci(target)
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

    prop = brahe.SGPPropagator.from_tle(line1, line2, step_size)
    base_epoch = prop.epoch

    def run():
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


def _force_model_from_params(params: dict) -> "brahe.ForceModelConfig":
    """Build a brahe ForceModelConfig matching the task parameter dict.

    The parameter vector layout is brahe's standard
    `[mass, drag_area, Cd, srp_area, Cr]`, so drag/SRP use parameter indices
    rather than fixed values — that way the same vector drives every step.
    """
    gravity = brahe.GravityConfiguration.spherical_harmonic(
        degree=params["gravity_degree"],
        order=params["gravity_order"],
    )

    third_body = None
    bodies = []
    if params.get("third_body_sun"):
        bodies.append(brahe.ThirdBody.SUN)
    if params.get("third_body_moon"):
        bodies.append(brahe.ThirdBody.MOON)
    if bodies:
        third_body = brahe.ThirdBodyConfiguration(
            ephemeris_source=brahe.EphemerisSource.DE440s,
            bodies=bodies,
        )

    drag = None
    if params.get("drag"):
        drag = brahe.DragConfiguration(
            model=brahe.AtmosphericModel.NRLMSISE00,
            area=brahe.ParameterSource.parameter_index(1),
            cd=brahe.ParameterSource.parameter_index(2),
        )

    srp = None
    if params.get("srp"):
        srp = brahe.SolarRadiationPressureConfiguration(
            area=brahe.ParameterSource.parameter_index(3),
            cr=brahe.ParameterSource.parameter_index(4),
            eclipse_model=brahe.EclipseModel.CONICAL,
        )

    needs_mass = drag is not None or srp is not None
    mass = brahe.ParameterSource.parameter_index(0) if needs_mass else None

    return brahe.ForceModelConfig(
        gravity=gravity,
        drag=drag,
        srp=srp,
        third_body=third_body,
        relativity=False,
        mass=mass,
    )


def _numerical_rk4_run(task_name: str, params: dict, iterations: int) -> TaskResult:
    """Shared driver for the RK4 force-model propagation tasks."""
    ensure_eop()
    if params.get("drag") and not brahe.get_global_sw_initialization():
        brahe.initialize_sw()

    jd = params["jd"]
    oe = np.array(params["elements_deg"])
    step_size = params["step_size"]
    n_steps = params["n_steps"]
    param_vec = np.array(params["params"])

    epc = brahe.Epoch.from_jd(jd, brahe.TimeSystem.UTC)
    cart = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)

    force_config = _force_model_from_params(params)
    prop_config = brahe.NumericalPropagationConfig(
        brahe.IntegrationMethod.RK4,
        brahe.IntegratorConfig.fixed_step(step_size),
        brahe.VariationalConfig(),
    )

    def run():
        prop = brahe.NumericalOrbitPropagator(
            epc, cart, prop_config, force_config, param_vec
        )
        prop.set_trajectory_mode(brahe.TrajectoryMode.DISABLED)
        results = []
        # Use step_by rather than propagate_to: with fixed-step RK4,
        # propagate_to() can hit a known float-drift path in brahe that
        # permanently shrinks dt_next when the target epoch isn't exactly
        # representable as a multiple of step_size. step_by avoids that
        # path entirely.
        for _ in range(n_steps):
            prop.step_by(step_size)
            state = prop.current_state()
            results.append(state.tolist())
        return results

    times, results = time_iterations(run, iterations)

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
            "frame": "GCRF",
            "integrator": "RK4",
            "step_size": step_size,
            "gravity": f"{params['gravity_degree']}x{params['gravity_order']}",
        },
    )


def numerical_rk4_grav5x5(params: dict, iterations: int) -> TaskResult:
    return _numerical_rk4_run("propagation.numerical_rk4_grav5x5", params, iterations)


def numerical_rk4_grav20x20_sun_moon(params: dict, iterations: int) -> TaskResult:
    return _numerical_rk4_run(
        "propagation.numerical_rk4_grav20x20_sun_moon", params, iterations
    )


def numerical_rk4_grav80x80_full(params: dict, iterations: int) -> TaskResult:
    return _numerical_rk4_run(
        "propagation.numerical_rk4_grav80x80_full", params, iterations
    )
