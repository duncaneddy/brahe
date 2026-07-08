"""Subprocess dispatcher for the bench_gpu_rust binary."""

from __future__ import annotations

import json
import subprocess

from benchmarks.gpu_comparison.config import REPO_ROOT, set_data_alignment_env
from benchmarks.gpu_comparison.results import CellResult, SkipReason
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


RUST_MANIFEST = (
    REPO_ROOT
    / "benchmarks"
    / "gpu_comparison"
    / "implementations"
    / "rust"
    / "Cargo.toml"
)
RUST_BINARY = RUST_MANIFEST.parent / "target" / "release" / "bench_gpu_rust"


def run_rust_cell(
    *,
    task: BatchTask,
    config: BatchConfig,
    batch_size: int,
    iterations: int,
    seed: int,
    per_cell_budget_s: float,
) -> CellResult:
    """Run one cell via the Rust subprocess and return its CellResult."""
    if not RUST_BINARY.exists():
        return CellResult.skipped(
            task=task.name,
            config=config.name,
            dtype=config.dtype,
            batch_size=batch_size,
            reason=SkipReason.BACKEND_ERROR,
            error_message=f"Rust binary not built at {RUST_BINARY}",
        )

    set_data_alignment_env()
    input_payload = {
        "task": task.name,
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup_iterations": task.warmup_iterations("rust"),
        "seed": seed,
        "params": task.generate_inputs(batch_size, seed),
    }

    try:
        proc = subprocess.run(
            [str(RUST_BINARY)],
            input=json.dumps(input_payload),
            capture_output=True,
            text=True,
            timeout=per_cell_budget_s + 30.0,
        )
    except subprocess.TimeoutExpired:
        return CellResult.skipped(
            task=task.name,
            config=config.name,
            dtype=config.dtype,
            batch_size=batch_size,
            reason=SkipReason.BUDGET_EXCEEDED,
        )

    if proc.returncode != 0:
        return CellResult.skipped(
            task=task.name,
            config=config.name,
            dtype=config.dtype,
            batch_size=batch_size,
            reason=SkipReason.BACKEND_ERROR,
            error_message=proc.stderr[:1000],
        )

    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return CellResult.skipped(
            task=task.name,
            config=config.name,
            dtype=config.dtype,
            batch_size=batch_size,
            reason=SkipReason.BACKEND_ERROR,
            error_message=f"JSON decode error: {e}; stdout was: {proc.stdout[:500]}",
        )

    return CellResult.ok_cell(
        task=task.name,
        config=config.name,
        dtype=config.dtype,
        batch_size=batch_size,
        times=output["times_seconds"],
        metadata={
            "backend_extra": output.get("metadata", {}) | {"task": output["task"]},
        },
    )
