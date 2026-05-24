"""Frame transformation benchmark tasks (GCRF↔ITRF)."""

from __future__ import annotations

import math

import numpy as np

from benchmarks.gpu_comparison.implementations import astrojax_kernels
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


_ALL_CONFIGS = [
    BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust"),
    BatchConfig(name="astrojax-cpu", dtype="f64", backend="astrojax-cpu"),
    BatchConfig(name="astrojax-gpu", dtype="f32", backend="astrojax-gpu"),
    BatchConfig(name="astrojax-multigpu", dtype="f32", backend="astrojax-multigpu"),
]

# Heavier per-call than coordinates (precession/nutation/polar motion), so ladder
# caps lower than the coord tasks. The astrojax builder constructs the
# batched Epoch directly from MJD floats as JAX arrays
# (see _build_batched_epoch_from_mjd) — no Python per-row loop, so large
# batches are cheap.
_LADDER = [1, 100, 1_000, 10_000, 100_000, 1_000_000]


class GcrfToItrfStateTask(BatchTask):
    name = "frames.gcrf_to_itrf"
    module = "frames"
    description = "Transform N (epoch, state6) pairs from GCRF to ITRF"
    configs = _ALL_CONFIGS

    def batch_sizes(self) -> list[int]:
        return _LADDER

    def generate_inputs(self, batch_size: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        R_EARTH = 6378137.0
        GM_EARTH = 3.986004418e14
        base_mjd = 60310.0  # 2024-01-01 UTC
        mjds = base_mjd + rng.uniform(0.0, 5 * 365.25, batch_size)
        a = R_EARTH + rng.uniform(400e3, 1500e3, batch_size)
        v = np.sqrt(GM_EARTH / a)
        nu = rng.uniform(0.0, 2 * np.pi, batch_size)
        states = np.empty((batch_size, 6), dtype=np.float64)
        states[:, 0] = a * np.cos(nu)
        states[:, 1] = a * np.sin(nu)
        states[:, 2] = 0.0
        states[:, 3] = -v * np.sin(nu)
        states[:, 4] = v * np.cos(nu)
        states[:, 5] = 0.0
        return {"mjd_utc": mjds.tolist(), "state_gcrf": states.tolist()}


def _jnp_dtype(dtype: str):
    import jax.numpy as jnp
    return jnp.float32 if dtype == "f32" else jnp.float64


def _build_batched_epoch_from_mjd(mjd_utc_list, dtype_str: str):
    """Construct a batched Epoch directly from MJD UTC floats.

    Astrojax's ``Epoch`` is a registered pytree with three leaves:
    ``_jd: int32`` (integer Julian date), ``_seconds: float`` (seconds
    within the JD), ``_kahan_c: float`` (Kahan compensation, 0 at init).
    We build the leaves as JAX arrays directly — no Python per-row loop —
    then create one Epoch whose leaves are batched arrays.

    Tested against the per-element ctor:
        MJD M → JD_full = M + 2400000.5
              _jd = floor(JD_full)
              _seconds = (JD_full - floor(JD_full)) * 86400
    """
    import jax.numpy as jnp
    from astrojax import Epoch

    jdtype = jnp.float64 if dtype_str == "f64" else jnp.float32
    mjd_arr = jnp.asarray(mjd_utc_list, dtype=jdtype)
    jd_full = mjd_arr + 2400000.5
    _jd = jnp.floor(jd_full).astype(jnp.int32)
    _seconds = ((jd_full - jnp.floor(jd_full)) * 86400.0).astype(jdtype)
    _kahan_c = jnp.zeros_like(_seconds)
    return Epoch._from_internal(_jd, _seconds, _kahan_c)


def _build_gcrf_to_itrf(task, batch_size, dtype, seed, devices):
    """Astrojax kernel: vmap over (epoch, state) keeping EOP static."""
    import jax
    import jax.numpy as jnp
    from astrojax.frames import state_gcrf_to_itrf

    from benchmarks.gpu_comparison.config import BRAHE_EOP_FILE
    from benchmarks.gpu_comparison.data_alignment import load_eop_for_astrojax

    eop = load_eop_for_astrojax(BRAHE_EOP_FILE)

    params = task.generate_inputs(batch_size, seed)
    states = jnp.array(params["state_gcrf"], dtype=_jnp_dtype(dtype))
    batched_epoch = _build_batched_epoch_from_mjd(params["mjd_utc"], dtype)

    fn = jax.vmap(state_gcrf_to_itrf, in_axes=(None, 0, 0))

    if len(devices) == 1 and hasattr(devices[0], "device_kind"):
        placed_states = jax.device_put(states, devices[0])
        placed_epochs = jax.device_put(batched_epoch, devices[0])
        compiled = jax.jit(fn, device=devices[0])
        return (lambda _: compiled(eop, placed_epochs, placed_states)), {}
    elif len(devices) == 1:
        return (lambda _: states), {}
    else:
        # multi-GPU: simplification — fall back to single-device for this task in v1.
        return (lambda _: jax.jit(fn)(eop, batched_epoch, states)), {}


astrojax_kernels.register("frames.gcrf_to_itrf", _build_gcrf_to_itrf)
