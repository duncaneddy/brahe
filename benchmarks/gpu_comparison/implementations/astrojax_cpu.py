"""Spawn-isolated dispatcher for the ``astrojax-cpu`` config.

JAX initialises its device list on first import and cannot host both CPU and
GPU in the same process. The astrojax-CPU benchmark therefore runs in a
``spawn``-ed Python child with ``JAX_PLATFORMS=cpu`` set on the child.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict
from typing import Any

from benchmarks.gpu_comparison.results import CellResult, SkipReason
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


def run_astrojax_cell_in_child(
    *,
    task: BatchTask,
    config: BatchConfig,
    batch_size: int,
    iterations: int,
    seed: int,
) -> CellResult:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(
        target=_child_entry,
        args=(q, task, config, batch_size, iterations, seed),
    )
    p.start()
    p.join()

    if not q.empty():
        kind, payload = q.get()
        if kind == "cell_dict":
            return _cell_from_dict(payload)
        if kind == "error":
            return CellResult.skipped(
                task=task.name,
                config=config.name,
                dtype=config.dtype,
                batch_size=batch_size,
                reason=SkipReason.BACKEND_ERROR,
                error_message=payload,
            )
    return CellResult.skipped(
        task=task.name,
        config=config.name,
        dtype=config.dtype,
        batch_size=batch_size,
        reason=SkipReason.BACKEND_ERROR,
        error_message=f"child process exited with code {p.exitcode} and no result",
    )


def _child_entry(q, task, config, batch_size, iterations, seed) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        from benchmarks.gpu_comparison.implementations.astrojax_gpu import (
            run_astrojax_cell_in_process,
        )

        cell = run_astrojax_cell_in_process(
            task=task,
            config=config,
            batch_size=batch_size,
            iterations=iterations,
            seed=seed,
            force_cpu=True,
        )
        q.put(("cell_dict", _cell_to_dict(cell)))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"))


def _cell_to_dict(cell: CellResult) -> dict[str, Any]:
    return asdict(cell)


def _cell_from_dict(d: dict[str, Any]) -> CellResult:
    return CellResult(**d)
