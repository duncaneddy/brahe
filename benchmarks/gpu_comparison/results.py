"""Dataclasses and JSON I/O for the GPU-comparison benchmark suite.

The runner produces one ``BenchmarkRun`` per invocation, containing one
``CellResult`` per (task, config, batch_size) cell. Cells can be ``ok`` (with
timing data) or ``skipped`` (with a ``SkipReason``).
"""

from __future__ import annotations

import enum
import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = "1"


class SkipReason(str, enum.Enum):
    BUDGET_EXCEEDED = "budget_exceeded"
    BUDGET_PROJECTED_EXCEEDED = "budget_projected_exceeded"
    BELOW_MULTIGPU_MIN_BATCH = "below_multigpu_min_batch"
    CONFIG_NOT_SUPPORTED_BY_TASK = "config_not_supported_by_task"
    GLOBAL_BUDGET_EXCEEDED = "global_budget_exceeded"
    BACKEND_ERROR = "backend_error"


@dataclass
class GPUInfo:
    index: int
    model: str
    memory_mb: int
    driver: str
    cuda_runtime: str


@dataclass
class SystemInfo:
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    ram_gb: int
    os: str
    python_version: str
    rust_version: str
    brahe_version: str
    brahe_git_sha: Optional[str]
    astrojax_version: str
    astrojax_git_sha: Optional[str]
    jax_version: str
    gpus: list[GPUInfo]
    rayon_threads: int


@dataclass
class SchedulingPolicy:
    per_cell_budget_s: float = 120.0
    global_run_budget_s: float = 3600.0
    iterations: int = 10


@dataclass
class CellResult:
    """Result of one (task, config, batch_size) cell."""

    task: str
    config: str
    dtype: str
    batch_size: int
    status: str  # "ok" | "skipped"

    # ok-only fields
    iterations: Optional[int] = None
    times_seconds: Optional[list[float]] = None
    mean_time_s: Optional[float] = None
    p50_time_s: Optional[float] = None
    p99_time_s: Optional[float] = None
    throughput_ops_per_sec: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # skipped-only fields
    skip_reason: Optional[str] = None
    projected_time_s: Optional[float] = None
    error_message: Optional[str] = None

    @classmethod
    def ok_cell(
        cls,
        *,
        task: str,
        config: str,
        dtype: str,
        batch_size: int,
        times: list[float],
        metadata: dict[str, Any],
    ) -> "CellResult":
        mean = statistics.fmean(times)
        sorted_times = sorted(times)
        return cls(
            task=task,
            config=config,
            dtype=dtype,
            batch_size=batch_size,
            status="ok",
            iterations=len(times),
            times_seconds=list(times),
            mean_time_s=mean,
            p50_time_s=sorted_times[len(sorted_times) // 2],
            p99_time_s=sorted_times[
                min(len(sorted_times) - 1, int(0.99 * len(sorted_times)))
            ],
            throughput_ops_per_sec=(batch_size / mean) if mean > 0 else float("inf"),
            speedup_vs_baseline=None,
            metadata=metadata,
        )

    @classmethod
    def skipped(
        cls,
        *,
        task: str,
        config: str,
        dtype: str,
        batch_size: int,
        reason: SkipReason,
        projected_time_s: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> "CellResult":
        return cls(
            task=task,
            config=config,
            dtype=dtype,
            batch_size=batch_size,
            status="skipped",
            skip_reason=reason.value,
            projected_time_s=projected_time_s,
            error_message=error_message,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Drop fields that are None to keep the JSON compact and per-status clean.
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class BenchmarkRun:
    run_id: str
    started_at: str
    finished_at: str
    seed: int
    iterations: int
    scheduling: SchedulingPolicy
    system: SystemInfo
    data_alignment: dict[str, str]
    cells: list[CellResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "seed": self.seed,
            "iterations": self.iterations,
            "scheduling": asdict(self.scheduling),
            "system": {
                **{k: v for k, v in asdict(self.system).items() if k != "gpus"},
                "gpus": [asdict(g) for g in self.system.gpus],
            },
            "data_alignment": dict(self.data_alignment),
            "cells": [c.to_dict() for c in self.cells],
        }

    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # The run_id is expected to be a UTC timestamp with colons replaced by dashes.
        path = output_dir / f"run_{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def compute_speedup_vs_baseline(
    cells: list[CellResult],
    baseline_config: str,
) -> None:
    """Mutate each ok cell's ``speedup_vs_baseline`` to its ratio against the
    baseline-config cell with the same (task, batch_size). Cells without a
    matching baseline (e.g. baseline skipped) get ``None``."""
    baseline_by_key: dict[tuple[str, int], float] = {}
    for c in cells:
        if c.status == "ok" and c.config == baseline_config:
            baseline_by_key[(c.task, c.batch_size)] = c.throughput_ops_per_sec  # type: ignore[assignment]
    for c in cells:
        if c.status != "ok":
            continue
        baseline = baseline_by_key.get((c.task, c.batch_size))
        if baseline is None or baseline == 0:
            c.speedup_vs_baseline = None
        else:
            c.speedup_vs_baseline = c.throughput_ops_per_sec / baseline  # type: ignore[operator]
