"""In-process astrojax dispatcher (single GPU + multi-GPU, also CPU if forced).

The runner uses this directly for the ``astrojax-gpu`` and
``astrojax-multigpu`` configs. The ``astrojax-cpu`` config goes through
``astrojax_cpu.run_astrojax_cell_in_child`` to avoid the JAX
device-conflict in the parent runner process.
"""

from __future__ import annotations

import time
from typing import Any

from benchmarks.gpu_comparison.implementations import astrojax_kernels
from benchmarks.gpu_comparison.results import CellResult, SkipReason
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


def run_astrojax_cell_in_process(
    *,
    task: BatchTask,
    config: BatchConfig,
    batch_size: int,
    iterations: int,
    seed: int,
    force_cpu: bool = False,
) -> CellResult:
    devices = _resolve_devices(config.backend, force_cpu=force_cpu)
    if devices is None:
        return CellResult.skipped(
            task=task.name, config=config.name, dtype=config.dtype,
            batch_size=batch_size, reason=SkipReason.BACKEND_ERROR,
            error_message=f"no devices available for backend {config.backend}",
        )

    try:
        builder = astrojax_kernels.get(task.name)
    except KeyError as e:
        return CellResult.skipped(
            task=task.name, config=config.name, dtype=config.dtype,
            batch_size=batch_size, reason=SkipReason.CONFIG_NOT_SUPPORTED_BY_TASK,
            error_message=str(e),
        )

    try:
        kernel, kernel_args = builder(task, batch_size, config.dtype, seed, devices)
    except Exception as e:
        return CellResult.skipped(
            task=task.name, config=config.name, dtype=config.dtype,
            batch_size=batch_size, reason=SkipReason.BACKEND_ERROR,
            error_message=f"kernel builder error: {e}",
        )

    warmups = task.warmup_iterations(config.backend)
    try:
        for _ in range(warmups):
            result = kernel(kernel_args)
            _block(result)
    except Exception as e:
        return CellResult.skipped(
            task=task.name, config=config.name, dtype=config.dtype,
            batch_size=batch_size, reason=SkipReason.BACKEND_ERROR,
            error_message=f"warmup error: {e}",
        )

    times: list[float] = []
    try:
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            result = kernel(kernel_args)
            _block(result)
            times.append((time.perf_counter_ns() - t0) / 1e9)
    except Exception as e:
        return CellResult.skipped(
            task=task.name, config=config.name, dtype=config.dtype,
            batch_size=batch_size, reason=SkipReason.BACKEND_ERROR,
            error_message=f"timed loop error: {e}",
        )

    return CellResult.ok_cell(
        task=task.name, config=config.name, dtype=config.dtype,
        batch_size=batch_size, times=times,
        metadata={
            "backend_extra": {
                "n_devices": len(devices),
                "device_kinds": [str(getattr(d, "device_kind", d)) for d in devices],
                "warmup_iterations": warmups,
            },
        },
    )


def _block(result: Any) -> None:
    """Force computation to complete: call ``.block_until_ready()`` on every
    JAX array in ``result``, walking tuples / lists / dicts recursively.

    Some kernels (notably SGP4, which returns ``(r, v)``) return structured
    outputs. Calling ``block_until_ready`` only on the top-level value is a
    no-op for tuples, which would cause the timed loop to measure just the
    async dispatch overhead rather than actual kernel time.
    """
    if isinstance(result, (tuple, list)):
        for x in result:
            _block(x)
        return
    if isinstance(result, dict):
        for x in result.values():
            _block(x)
        return
    blocker = getattr(result, "block_until_ready", None)
    if callable(blocker):
        blocker()


def _resolve_devices(backend: str, *, force_cpu: bool = False) -> list | None:
    """Return JAX devices for the backend. None if no devices available.

    For non-JAX callers (the ping test), returns a sentinel list so the
    dispatcher proceeds with a no-op device pick.
    """
    if force_cpu or backend == "astrojax-cpu":
        try:
            import jax
            return jax.devices("cpu")
        except (ImportError, RuntimeError):
            return ["cpu-sentinel"]
    try:
        import jax
        gpus = jax.devices("gpu")
        if not gpus:
            return None
        if backend == "astrojax-gpu":
            return [gpus[0]]
        if backend == "astrojax-multigpu":
            return gpus
        return None
    except (ImportError, RuntimeError):
        return None
