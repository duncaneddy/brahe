"""Time-system conversion benchmark tasks.

Astrojax does not expose `utc_to_tt` / `utc_to_tai` / etc. as public callables
(per spike 02). It does expose the building blocks: `leap_seconds_tai_utc`
table-lookup and the `TT_TAI = 32.184` constant. This module benchmarks the
composed conversion ``MJD UTC → MJD TT`` end-to-end on both sides, with the
astrojax side built from those primitives.
"""

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

_LADDER = [1, 100, 10_000, 1_000_000, 10_000_000, 100_000_000]

# 2000-01-01 UTC = MJD 51544.0; sample across 30 years
_MJD_BASE = 51544.0
_MJD_SPAN_DAYS = 365.25 * 30.0


class UtcMjdToTtMjdTask(BatchTask):
    name = "time.utc_mjd_to_tt_mjd"
    module = "time"
    description = "Convert N MJD epochs from UTC to TT (UTC + leap_seconds + 32.184)"
    configs = _ALL_CONFIGS

    def batch_sizes(self) -> list[int]:
        return _LADDER

    def generate_inputs(self, batch_size: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        mjds = _MJD_BASE + rng.uniform(0.0, _MJD_SPAN_DAYS, batch_size)
        return {"mjd_utc": mjds.tolist()}


def _jnp_dtype(dtype: str):
    import jax.numpy as jnp
    return jnp.float32 if dtype == "f32" else jnp.float64


def _build_utc_mjd_to_tt_mjd(task, batch_size, dtype, seed, devices):
    import jax
    import jax.numpy as jnp
    from astrojax.time import TT_TAI, leap_seconds_tai_utc

    SECONDS_PER_DAY = 86400.0

    def utc_mjd_to_tt_mjd(mjd_utc):
        return mjd_utc + (leap_seconds_tai_utc(mjd_utc) + TT_TAI) / SECONDS_PER_DAY

    params = task.generate_inputs(batch_size, seed)
    mjds = jnp.array(params["mjd_utc"], dtype=_jnp_dtype(dtype))

    if len(devices) == 1 and hasattr(devices[0], "device_kind"):
        placed = jax.device_put(mjds, devices[0])
        compiled = jax.jit(jax.vmap(utc_mjd_to_tt_mjd), device=devices[0])
        return (lambda _: compiled(placed)), {}
    elif len(devices) == 1:
        return (lambda _: mjds), {}
    else:
        n_dev = len(devices)
        padded = ((batch_size + n_dev - 1) // n_dev) * n_dev
        if padded > batch_size:
            pad = jnp.zeros(padded - batch_size, dtype=mjds.dtype)
            mjds = jnp.concatenate([mjds, pad])
        reshaped = mjds.reshape(n_dev, -1)
        placed = shard_across_devices(reshaped, devices)
        compiled = jax.pmap(jax.vmap(utc_mjd_to_tt_mjd))
        return (lambda _: compiled(placed)), {}


astrojax_kernels.register("time.utc_mjd_to_tt_mjd", _build_utc_mjd_to_tt_mjd)
