"""Full force model propagation benchmark tasks.

For v1 we ship one task — 5x5 spherical-harmonic gravity over one LEO orbit —
across all four backends. Larger tasks (20x20, 20x20+drag+SRP+third-body)
can be added by following the same pattern; they need EGM2008 coefficient
loading on the astrojax side, which is non-trivial to verify against the
brahe-bundled file.
"""

from __future__ import annotations

import math

import numpy as np

from benchmarks.gpu_comparison.implementations import astrojax_kernels
from benchmarks.gpu_comparison.implementations.jax_utils import shard_across_devices
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask

R_EARTH = 6378137.0
GM_EARTH = 3.986004418e14


class ForceModelGrav5x5Task(BatchTask):
    name = "force_model.grav_5x5"
    module = "force_model"
    description = (
        "RK4-propagate N LEO orbits over ~1 orbital period (30s step) with "
        "5x5 spherical-harmonic gravity (EGM2008)."
    )
    configs = [
        BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust"),
        BatchConfig(name="astrojax-cpu", dtype="f64", backend="astrojax-cpu"),
        BatchConfig(name="astrojax-gpu", dtype="f32", backend="astrojax-gpu"),
        BatchConfig(name="astrojax-multigpu", dtype="f32", backend="astrojax-multigpu"),
    ]

    def multigpu_min_batch(self) -> int:
        # 100x lower than the default 100k since each orbit is 180 RK4 steps
        # with 5x5 SH gravity — already heavy work per element. 1k orbits
        # split across 2 A100s is a meaningful workload.
        return 1_000

    STEP_SIZE = 30.0
    N_STEPS = 180

    def batch_sizes(self) -> list[int]:
        # Extended past the original [1, 10, 100, 1000] cap to find the
        # GPU crossover. Astrojax's RK4+SH(5x5) graph is heavy per element
        # at small batches but should amortise at 10k+; we sweep up to 100k
        # so the runner sees the crossover. The scheduler skips cells that
        # would blow the per-cell budget.
        return [1, 10, 100, 1_000, 10_000, 100_000]

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
            "gravity_degree": 5,
            "gravity_order": 5,
        }


def _jnp_dtype(dtype: str):
    import jax.numpy as jnp
    return jnp.float32 if dtype == "f32" else jnp.float64


def _build_grav_5x5(task, batch_size, dtype, seed, devices):
    import jax
    import jax.numpy as jnp
    from astrojax import Epoch, set_dtype
    from astrojax.eop import zero_eop
    from astrojax.integrators import rk4_step
    from astrojax.orbit_dynamics.config import ForceModelConfig
    from astrojax.orbit_dynamics.factory import create_orbit_dynamics
    from astrojax.orbit_dynamics.gravity import GravityModel

    # Align astrojax's internal dtype with the config's requested dtype.
    set_dtype(jnp.float32 if dtype == "f32" else jnp.float64)

    grav = GravityModel.from_type("JGM3")
    epoch_0 = Epoch(2024, 6, 15, 12, 0, 0.0)
    cfg = ForceModelConfig(
        gravity_type="spherical_harmonics",
        gravity_model=grav,
        gravity_degree=5,
        gravity_order=5,
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
        # Multi-GPU: shard the batch across devices. The dynamics closure
        # (gravity model, force-model config) is captured in `_propagate_one`
        # and JIT-compiled per device; only the per-orbit initial states are
        # sharded.
        n_dev = len(devices)
        padded = ((batch_size + n_dev - 1) // n_dev) * n_dev
        if padded > batch_size:
            pad = jnp.zeros((padded - batch_size, 6), dtype=states.dtype)
            states = jnp.concatenate([states, pad], axis=0)
        reshaped = states.reshape(n_dev, -1, 6)
        placed = shard_across_devices(reshaped, devices)
        compiled = jax.pmap(jax.vmap(_propagate_one))
        return (lambda _: compiled(placed)), {}


astrojax_kernels.register("force_model.grav_5x5", _build_grav_5x5)
