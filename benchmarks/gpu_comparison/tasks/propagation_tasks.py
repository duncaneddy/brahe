"""SGP4 and numerical propagation benchmark tasks."""

from __future__ import annotations

import math

import numpy as np

from benchmarks.gpu_comparison.implementations import astrojax_kernels
from benchmarks.gpu_comparison.implementations.jax_utils import shard_across_devices
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


_ALL_CONFIGS = [
    BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust"),
    BatchConfig(name="astrojax-cpu", dtype="f64", backend="astrojax-cpu"),
    BatchConfig(name="astrojax-gpu", dtype="f32", backend="astrojax-gpu"),
    BatchConfig(name="astrojax-multigpu", dtype="f32", backend="astrojax-multigpu"),
]

# Well-known ISS TLE used by the existing astrojax / brahe benchmarks.
ISS_TLE_LINE1 = "1 25544U 98067A   24127.82853009  .00015698  00000+0  27310-3 0  9995"
ISS_TLE_LINE2 = "2 25544  51.6393 160.4574 0003580 140.6673 205.7250 15.50957674452123"


class Sgp4IssSweepTask(BatchTask):
    name = "propagation.sgp4_iss_sweep"
    module = "propagation"
    description = "Propagate ISS TLE to N time offsets (in minutes since epoch) using SGP4"
    configs = _ALL_CONFIGS

    def batch_sizes(self) -> list[int]:
        return [1, 100, 1_000, 10_000, 100_000, 1_000_000]

    def generate_inputs(self, batch_size: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        tsince = rng.uniform(0.0, 7.0 * 24.0 * 60.0, batch_size)
        return {
            "line1": ISS_TLE_LINE1,
            "line2": ISS_TLE_LINE2,
            "tsince_minutes": tsince.tolist(),
        }


# ──────────────────────────── Numerical two-body / J2 ────────────────────────────

R_EARTH = 6378137.0
GM_EARTH = 3.986004418e14


class NumericalTwobodyJ2Task(BatchTask):
    name = "propagation.numerical_twobody_j2"
    module = "propagation"
    description = (
        "RK4-propagate N LEO orbits over ~1 orbital period (30s step) with J2-only "
        "spherical-harmonic gravity (degree=2, order=0)"
    )
    # astrojax integrators+orbit_dynamics path is heavier — restrict to brahe + astrojax-gpu
    # to keep iteration count manageable on CPU.
    configs = [
        BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust"),
        BatchConfig(name="astrojax-cpu", dtype="f64", backend="astrojax-cpu"),
        BatchConfig(name="astrojax-gpu", dtype="f32", backend="astrojax-gpu"),
        BatchConfig(name="astrojax-multigpu", dtype="f32", backend="astrojax-multigpu"),
    ]

    STEP_SIZE = 30.0  # seconds
    N_STEPS = 180  # 90 minutes ~ one LEO period

    def batch_sizes(self) -> list[int]:
        return [1, 10, 100, 1_000, 10_000]

    def multigpu_min_batch(self) -> int:
        # 100x lower than the default 100k since 180-step RK4 propagation is
        # heavy enough per orbit that 1k orbits split across 2 A100s is a
        # meaningful workload — and the suite's ladder caps at 10k anyway.
        return 1_000

    def generate_inputs(self, batch_size: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        a = R_EARTH + rng.uniform(400e3, 800e3, batch_size)
        v = np.sqrt(GM_EARTH / a)
        nu = rng.uniform(0.0, 2 * np.pi, batch_size)
        states = np.empty((batch_size, 6), dtype=np.float64)
        states[:, 0] = a * np.cos(nu)
        states[:, 1] = a * np.sin(nu)
        states[:, 2] = 0.0
        states[:, 3] = -v * np.sin(nu)
        states[:, 4] = v * np.cos(nu)
        states[:, 5] = 0.0
        return {
            "states_eci": states.tolist(),
            "step_size": self.STEP_SIZE,
            "n_steps": self.N_STEPS,
        }


# ──────────────────────────── astrojax kernel builders ────────────────────────────


def _jnp_dtype(dtype: str):
    import jax.numpy as jnp
    return jnp.float32 if dtype == "f32" else jnp.float64


def _build_sgp4_iss_sweep(task, batch_size, dtype, seed, devices):
    import jax
    import jax.numpy as jnp
    from astrojax.sgp4 import create_sgp4_propagator

    params = task.generate_inputs(batch_size, seed)
    _, prop_fn = create_sgp4_propagator(params["line1"], params["line2"], gravity="wgs72")
    times = jnp.array(params["tsince_minutes"], dtype=_jnp_dtype(dtype))

    if len(devices) == 1 and hasattr(devices[0], "device_kind"):
        placed = jax.device_put(times, devices[0])
        compiled = jax.jit(jax.vmap(prop_fn), device=devices[0])
        return (lambda _: compiled(placed)), {}
    elif len(devices) == 1:
        return (lambda _: times), {}
    else:
        n_dev = len(devices)
        padded = ((batch_size + n_dev - 1) // n_dev) * n_dev
        if padded > batch_size:
            pad = jnp.zeros(padded - batch_size, dtype=times.dtype)
            times = jnp.concatenate([times, pad])
        reshaped = times.reshape(n_dev, -1)
        placed = shard_across_devices(reshaped, devices)
        compiled = jax.pmap(jax.vmap(prop_fn))
        return (lambda _: compiled(placed)), {}


def _build_numerical_twobody_j2(task, batch_size, dtype, seed, devices):
    """RK4 propagation with J2-only gravity for N initial states."""
    import jax
    import jax.numpy as jnp

    # Build the dynamics once; reuse for all batch members.
    from astrojax import Epoch, set_dtype
    from astrojax.eop import zero_eop
    from astrojax.integrators import rk4_step
    from astrojax.orbit_dynamics.config import ForceModelConfig
    from astrojax.orbit_dynamics.factory import create_orbit_dynamics
    from astrojax.orbit_dynamics.gravity import GravityModel

    # Align astrojax's internal dtype with the config's requested dtype so the
    # rk4 scan body's carry input/output dtypes match.
    set_dtype(jnp.float32 if dtype == "f32" else jnp.float64)

    grav = GravityModel.from_type("JGM3")
    epoch_0 = Epoch(2024, 6, 15, 12, 0, 0.0)
    cfg = ForceModelConfig(
        gravity_type="spherical_harmonics",
        gravity_model=grav,
        gravity_degree=2,
        gravity_order=0,
    )
    dyn = create_orbit_dynamics(zero_eop(), epoch_0, cfg)

    dt = task.STEP_SIZE
    n_steps = task.N_STEPS
    params = task.generate_inputs(batch_size, seed)
    states = jnp.array(params["states_eci"], dtype=_jnp_dtype(dtype))

    def _propagate_one(x0):
        def body(x, _):
            return rk4_step(dyn, 0.0, x, dt).state, None

        final, _ = jax.lax.scan(body, x0, None, length=n_steps)
        return final

    if len(devices) == 1 and hasattr(devices[0], "device_kind"):
        placed = jax.device_put(states, devices[0])
        compiled = jax.jit(jax.vmap(_propagate_one), device=devices[0])
        return (lambda _: compiled(placed)), {}
    elif len(devices) == 1:
        return (lambda _: states), {}
    else:
        n_dev = len(devices)
        padded = ((batch_size + n_dev - 1) // n_dev) * n_dev
        if padded > batch_size:
            pad = jnp.zeros((padded - batch_size, 6), dtype=states.dtype)
            states = jnp.concatenate([states, pad], axis=0)
        reshaped = states.reshape(n_dev, -1, 6)
        placed = shard_across_devices(reshaped, devices)
        compiled = jax.pmap(jax.vmap(_propagate_one))
        return (lambda _: compiled(placed)), {}


astrojax_kernels.register("propagation.sgp4_iss_sweep", _build_sgp4_iss_sweep)
astrojax_kernels.register("propagation.numerical_twobody_j2", _build_numerical_twobody_j2)
