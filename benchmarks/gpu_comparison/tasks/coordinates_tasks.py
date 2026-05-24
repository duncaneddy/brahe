"""Coordinate transformation benchmark tasks."""

from __future__ import annotations

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

_LADDER = [1, 100, 10_000, 100_000, 1_000_000, 10_000_000]


class GeodeticToEcefTask(BatchTask):
    name = "coordinates.geodetic_to_ecef"
    module = "coordinates"
    description = "Convert N (lon°, lat°, alt) triples to ECEF Cartesian"
    configs = _ALL_CONFIGS

    def batch_sizes(self) -> list[int]:
        return _LADDER

    def generate_inputs(self, batch_size: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        pts = np.empty((batch_size, 3), dtype=np.float64)
        pts[:, 0] = rng.uniform(-180.0, 180.0, batch_size)
        pts[:, 1] = rng.uniform(-89.0, 89.0, batch_size)
        pts[:, 2] = rng.uniform(0.0, 1.0e6, batch_size)
        return {"points": pts.tolist()}


class KeplerianToCartesianTask(BatchTask):
    name = "coordinates.keplerian_to_cartesian"
    module = "coordinates"
    description = "Convert N Keplerian element sets [a, e, i, RAAN, omega, M] (degrees) to ECI Cartesian"
    configs = _ALL_CONFIGS

    def batch_sizes(self) -> list[int]:
        return _LADDER

    def generate_inputs(self, batch_size: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        R_EARTH = 6378137.0
        oes = np.empty((batch_size, 6), dtype=np.float64)
        oes[:, 0] = R_EARTH + rng.uniform(400e3, 36000e3, batch_size)
        oes[:, 1] = rng.uniform(0.001, 0.3, batch_size)
        oes[:, 2] = rng.uniform(0.0, 180.0, batch_size)
        oes[:, 3] = rng.uniform(0.0, 360.0, batch_size)
        oes[:, 4] = rng.uniform(0.0, 360.0, batch_size)
        oes[:, 5] = rng.uniform(0.0, 360.0, batch_size)
        return {"elements": oes.tolist()}


class EnzToAzelTask(BatchTask):
    name = "coordinates.enz_to_azel"
    module = "coordinates"
    description = "Convert N topocentric ENZ vectors to (azimuth°, elevation°, range)"
    configs = _ALL_CONFIGS

    def batch_sizes(self) -> list[int]:
        return _LADDER

    def generate_inputs(self, batch_size: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        vecs = np.empty((batch_size, 3), dtype=np.float64)
        vecs[:, 0] = rng.uniform(-1000e3, 1000e3, batch_size)
        vecs[:, 1] = rng.uniform(-1000e3, 1000e3, batch_size)
        vecs[:, 2] = rng.uniform(1.0, 1000e3, batch_size)
        return {"vectors": vecs.tolist()}


# ──────────────────────────── astrojax kernel builders ────────────────────────────


def _jnp_dtype(dtype: str):
    import jax.numpy as jnp
    return jnp.float32 if dtype == "f32" else jnp.float64


def _wrap_kernel(kernel, args, devices):
    """Wrap a vmapped+jitted kernel into a (callable, args) pair for the dispatcher."""
    import jax

    if len(devices) == 1:
        device = devices[0]
        if hasattr(device, "device_kind"):
            placed = jax.device_put(args, device)
        else:
            placed = args
        compiled = jax.jit(kernel, device=device) if hasattr(device, "device_kind") else jax.jit(kernel)
        return (lambda _: compiled(placed)), {}
    else:
        # multi-GPU: shard input batch across devices
        n_dev = len(devices)
        batch = args.shape[0]
        padded = ((batch + n_dev - 1) // n_dev) * n_dev
        if padded > batch:
            import jax.numpy as jnp
            pad = jnp.zeros((padded - batch,) + args.shape[1:], dtype=args.dtype)
            args = jnp.concatenate([args, pad], axis=0)
        sharded_shape = (n_dev, padded // n_dev) + args.shape[1:]
        reshaped = args.reshape(sharded_shape)
        placed = shard_across_devices(reshaped, devices)
        compiled = jax.pmap(jax.vmap(kernel))
        return (lambda _: compiled(placed)), {}


def _build_geodetic_to_ecef(task, batch_size, dtype, seed, devices):
    import jax
    import jax.numpy as jnp
    from astrojax.coordinates import position_geodetic_to_ecef

    params = task.generate_inputs(batch_size, seed)
    pts = jnp.array(params["points"], dtype=_jnp_dtype(dtype))
    # Plan's input is in degrees; astrojax default is radians; convert.
    pts_rad = pts.at[:, 0].multiply(jnp.pi / 180.0).at[:, 1].multiply(jnp.pi / 180.0)

    if len(devices) == 1 and hasattr(devices[0], "device_kind"):
        placed = jax.device_put(pts_rad, devices[0])
        compiled = jax.jit(jax.vmap(position_geodetic_to_ecef), device=devices[0])
        return (lambda _: compiled(placed)), {}
    elif len(devices) == 1:
        return (lambda _: pts_rad), {}  # sentinel fallback for tests
    else:
        n_dev = len(devices)
        batch = pts_rad.shape[0]
        padded = ((batch + n_dev - 1) // n_dev) * n_dev
        if padded > batch:
            pad = jnp.zeros((padded - batch, 3), dtype=pts_rad.dtype)
            pts_rad = jnp.concatenate([pts_rad, pad], axis=0)
        reshaped = pts_rad.reshape(n_dev, -1, 3)
        placed = shard_across_devices(reshaped, devices)
        compiled = jax.pmap(jax.vmap(position_geodetic_to_ecef))
        return (lambda _: compiled(placed)), {}


def _build_keplerian_to_cartesian(task, batch_size, dtype, seed, devices):
    import jax
    import jax.numpy as jnp
    from astrojax.coordinates import state_koe_to_eci

    params = task.generate_inputs(batch_size, seed)
    oes_deg = jnp.array(params["elements"], dtype=_jnp_dtype(dtype))
    # Plan generates degrees; astrojax default is radians for i, RAAN, omega, M.
    deg_to_rad = jnp.pi / 180.0
    oes = oes_deg.at[:, 2:].multiply(deg_to_rad)

    if len(devices) == 1 and hasattr(devices[0], "device_kind"):
        placed = jax.device_put(oes, devices[0])
        compiled = jax.jit(jax.vmap(state_koe_to_eci), device=devices[0])
        return (lambda _: compiled(placed)), {}
    elif len(devices) == 1:
        return (lambda _: oes), {}
    else:
        n_dev = len(devices)
        batch = oes.shape[0]
        padded = ((batch + n_dev - 1) // n_dev) * n_dev
        if padded > batch:
            pad = jnp.zeros((padded - batch, 6), dtype=oes.dtype)
            oes = jnp.concatenate([oes, pad], axis=0)
        reshaped = oes.reshape(n_dev, -1, 6)
        placed = shard_across_devices(reshaped, devices)
        compiled = jax.pmap(jax.vmap(state_koe_to_eci))
        return (lambda _: compiled(placed)), {}


def _build_enz_to_azel(task, batch_size, dtype, seed, devices):
    import jax
    import jax.numpy as jnp
    from astrojax.coordinates import position_enz_to_azel

    params = task.generate_inputs(batch_size, seed)
    enzs = jnp.array(params["vectors"], dtype=_jnp_dtype(dtype))

    if len(devices) == 1 and hasattr(devices[0], "device_kind"):
        placed = jax.device_put(enzs, devices[0])
        compiled = jax.jit(jax.vmap(position_enz_to_azel), device=devices[0])
        return (lambda _: compiled(placed)), {}
    elif len(devices) == 1:
        return (lambda _: enzs), {}
    else:
        n_dev = len(devices)
        batch = enzs.shape[0]
        padded = ((batch + n_dev - 1) // n_dev) * n_dev
        if padded > batch:
            pad = jnp.zeros((padded - batch, 3), dtype=enzs.dtype)
            enzs = jnp.concatenate([enzs, pad], axis=0)
        reshaped = enzs.reshape(n_dev, -1, 3)
        placed = shard_across_devices(reshaped, devices)
        compiled = jax.pmap(jax.vmap(position_enz_to_azel))
        return (lambda _: compiled(placed)), {}


astrojax_kernels.register("coordinates.geodetic_to_ecef", _build_geodetic_to_ecef)
astrojax_kernels.register("coordinates.keplerian_to_cartesian", _build_keplerian_to_cartesian)
astrojax_kernels.register("coordinates.enz_to_azel", _build_enz_to_azel)
