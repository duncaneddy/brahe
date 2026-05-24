"""Top-level orchestrator: per (task × config) sweep the batch ladder.

This module is intentionally framework-agnostic; the typer CLI in
``__main__.py`` calls into it.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Optional

from benchmarks.gpu_comparison.config import (
    RESULTS_DIR,
    collect_system_info,
    data_alignment_record,
    set_data_alignment_env,
)
from benchmarks.gpu_comparison.implementations.astrojax_cpu import (
    run_astrojax_cell_in_child,
)
from benchmarks.gpu_comparison.implementations.astrojax_gpu import (
    run_astrojax_cell_in_process,
)
from benchmarks.gpu_comparison.implementations.rust_backend import run_rust_cell
from benchmarks.gpu_comparison.registry import filter_tasks
from benchmarks.gpu_comparison.results import (
    BenchmarkRun,
    CellResult,
    SchedulingPolicy,
    compute_speedup_vs_baseline,
)
from benchmarks.gpu_comparison.scheduler import schedule_ladder
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


BASELINE_CONFIG = "brahe-rust-rayon"


def run_one_cell(
    task: BatchTask,
    config: BatchConfig,
    batch_size: int,
    iterations: int,
    seed: int,
    per_cell_budget_s: float,
) -> CellResult:
    """Dispatch one cell to the appropriate backend."""
    if config.backend == "rust":
        return run_rust_cell(
            task=task, config=config, batch_size=batch_size,
            iterations=iterations, seed=seed,
            per_cell_budget_s=per_cell_budget_s,
        )
    if config.backend == "astrojax-cpu":
        return run_astrojax_cell_in_child(
            task=task, config=config, batch_size=batch_size,
            iterations=iterations, seed=seed,
        )
    if config.backend in ("astrojax-gpu", "astrojax-multigpu"):
        return run_astrojax_cell_in_process(
            task=task, config=config, batch_size=batch_size,
            iterations=iterations, seed=seed,
        )
    raise ValueError(f"unknown backend {config.backend!r}")


def run_one_task(
    task: BatchTask,
    *,
    iterations: int,
    seed: int,
    per_cell_budget_s: float,
    configs_filter: Optional[list[str]],
    progress: bool = True,
) -> list[CellResult]:
    """Sweep every (config, batch_size) for one task."""
    policy = SchedulingPolicy(per_cell_budget_s=per_cell_budget_s, iterations=iterations)
    cells: list[CellResult] = []
    for cfg in task.configs:
        if configs_filter is not None and cfg.name not in configs_filter:
            continue

        def _with_progress(b: int, c=cfg) -> CellResult:
            if progress:
                import sys
                print(f"  [{task.name}] {c.name} batch={b} ...", flush=True, file=sys.stderr)
            t0 = time.time()
            cell = run_one_cell(task, c, b, iterations, seed, per_cell_budget_s)
            dt = time.time() - t0
            if progress:
                import sys
                if cell.status == "ok":
                    print(
                        f"    -> ok  mean={cell.mean_time_s:.6f}s "
                        f"thr={cell.throughput_ops_per_sec:.3e} ops/s  ({dt:.1f}s wall)",
                        flush=True, file=sys.stderr,
                    )
                else:
                    print(
                        f"    -> skipped: {cell.skip_reason}  ({dt:.1f}s wall)",
                        flush=True, file=sys.stderr,
                    )
            return cell

        cells.extend(
            schedule_ladder(
                task=task, config=cfg,
                run_one_cell=_with_progress,
                policy=policy,
            )
        )
    return cells


def run_suite(
    *,
    module: Optional[str] = None,
    task_name: Optional[str] = None,
    configs_filter: Optional[list[str]] = None,
    iterations: int = 10,
    seed: int = 42,
    per_cell_budget_s: float = 120.0,
    global_run_budget_s: float = 3600.0,
    output_dir: Path = RESULTS_DIR,
) -> Path:
    set_data_alignment_env()

    import benchmarks.gpu_comparison.tasks.register_all  # noqa: F401 — populates registry

    tasks = filter_tasks(module=module, task_name=task_name)
    if not tasks:
        raise ValueError("no tasks match the requested filters")

    import sys

    started_at = dt.datetime.now(dt.timezone.utc)
    run_id = started_at.strftime("%Y-%m-%dT%H-%M-%SZ")
    cells: list[CellResult] = []
    system = collect_system_info()
    data_align = data_alignment_record()

    def _save_partial() -> Path:
        """Save a partial run file so a hang doesn't lose progress."""
        compute_speedup_vs_baseline(cells, baseline_config=BASELINE_CONFIG)
        run = BenchmarkRun(
            run_id=run_id,
            started_at=started_at.isoformat(),
            finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            seed=seed, iterations=iterations,
            scheduling=SchedulingPolicy(
                per_cell_budget_s=per_cell_budget_s,
                global_run_budget_s=global_run_budget_s,
                iterations=iterations,
            ),
            system=system, data_alignment=data_align, cells=cells,
        )
        return run.save(output_dir)

    for i, t in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] task: {t.name}", flush=True, file=sys.stderr)
        cells.extend(run_one_task(
            t,
            iterations=iterations, seed=seed,
            per_cell_budget_s=per_cell_budget_s,
            configs_filter=configs_filter,
        ))
        # Persist progress after each task so a hang doesn't lose everything.
        path = _save_partial()
        print(f"  partial results -> {path}", flush=True, file=sys.stderr)

    return _save_partial()
