"""Per-(task, config) batch-ladder iteration with wall-clock budget."""

from __future__ import annotations

from typing import Callable

from benchmarks.gpu_comparison.results import (
    CellResult,
    SchedulingPolicy,
    SkipReason,
)
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


def should_run_multigpu(task: BatchTask, batch_size: int) -> bool:
    return batch_size >= task.multigpu_min_batch()


def project_next_cell_time(
    *,
    prev_cell: CellResult,
    prev_batch: int,
    next_batch: int,
    iterations: int,
) -> float:
    """Linear-scaling projection of next cell's total wall time in seconds."""
    if prev_cell.mean_time_s is None or prev_batch == 0:
        return 0.0
    per_iter_at_next = prev_cell.mean_time_s * (next_batch / prev_batch)
    return iterations * per_iter_at_next


def schedule_ladder(
    *,
    task: BatchTask,
    config: BatchConfig,
    run_one_cell: Callable[[int], CellResult],
    policy: SchedulingPolicy,
) -> list[CellResult]:
    """Sweep ``task.batch_sizes()`` for ``config``, respecting the per-cell
    wall-clock budget. ``run_one_cell(batch_size)`` executes one cell."""

    ladder = task.batch_sizes()
    cells: list[CellResult] = []
    is_multigpu = config.backend == "astrojax-multigpu"
    prev_ok: CellResult | None = None
    prev_batch: int | None = None
    budget_exhausted = False

    for batch_size in ladder:
        if is_multigpu and not should_run_multigpu(task, batch_size):
            cells.append(
                CellResult.skipped(
                    task=task.name,
                    config=config.name,
                    dtype=config.dtype,
                    batch_size=batch_size,
                    reason=SkipReason.BELOW_MULTIGPU_MIN_BATCH,
                )
            )
            continue

        if budget_exhausted:
            cells.append(
                CellResult.skipped(
                    task=task.name,
                    config=config.name,
                    dtype=config.dtype,
                    batch_size=batch_size,
                    reason=SkipReason.BUDGET_PROJECTED_EXCEEDED,
                )
            )
            continue

        if prev_ok is not None and prev_batch is not None:
            projected = project_next_cell_time(
                prev_cell=prev_ok,
                prev_batch=prev_batch,
                next_batch=batch_size,
                iterations=policy.iterations,
            )
            if projected > policy.per_cell_budget_s:
                cells.append(
                    CellResult.skipped(
                        task=task.name,
                        config=config.name,
                        dtype=config.dtype,
                        batch_size=batch_size,
                        reason=SkipReason.BUDGET_PROJECTED_EXCEEDED,
                        projected_time_s=projected,
                    )
                )
                budget_exhausted = True
                continue

        cell = run_one_cell(batch_size)
        cells.append(cell)
        if cell.status == "ok":
            prev_ok = cell
            prev_batch = batch_size

    return cells
