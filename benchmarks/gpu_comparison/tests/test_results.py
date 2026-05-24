import json
from pathlib import Path

import pytest

from benchmarks.gpu_comparison.results import (
    BenchmarkRun,
    CellResult,
    SchedulingPolicy,
    SkipReason,
    SystemInfo,
    compute_speedup_vs_baseline,
)


def _make_system() -> SystemInfo:
    return SystemInfo(
        cpu_model="Test CPU",
        cpu_physical_cores=4,
        cpu_logical_cores=8,
        ram_gb=16,
        os="Linux 6.0",
        python_version="3.12.0",
        rust_version="rustc 1.85.0",
        brahe_version="1.5.2",
        brahe_git_sha="deadbeef",
        astrojax_version="0.8.0",
        astrojax_git_sha=None,
        jax_version="0.9.0",
        gpus=[],
        rayon_threads=4,
    )


def test_cell_result_ok_serialises():
    cell = CellResult(
        task="t",
        config="brahe-rust-rayon",
        dtype="f64",
        batch_size=1000,
        status="ok",
        iterations=10,
        times_seconds=[0.001, 0.0011, 0.0009],
        mean_time_s=0.001,
        p50_time_s=0.001,
        p99_time_s=0.0011,
        throughput_ops_per_sec=1_000_000.0,
        speedup_vs_baseline=1.0,
        metadata={"backend_extra": {"rayon_threads": 4}},
    )
    d = cell.to_dict()
    assert d["status"] == "ok"
    assert d["batch_size"] == 1000
    assert d["mean_time_s"] == 0.001


def test_cell_result_skipped_omits_timing_fields():
    cell = CellResult.skipped(
        task="t",
        config="brahe-rust-rayon",
        dtype="f64",
        batch_size=1_000_000,
        reason=SkipReason.BUDGET_PROJECTED_EXCEEDED,
        projected_time_s=1240.0,
    )
    d = cell.to_dict()
    assert d["status"] == "skipped"
    assert d["skip_reason"] == "budget_projected_exceeded"
    assert d["projected_time_s"] == 1240.0
    assert "times_seconds" not in d


def test_compute_speedup_vs_baseline():
    base = CellResult.ok_cell(task="t", config="brahe-rust-rayon", dtype="f64",
                              batch_size=1000, times=[0.01] * 10, metadata={})
    candidate = CellResult.ok_cell(task="t", config="astrojax-gpu", dtype="f32",
                                   batch_size=1000, times=[0.001] * 10, metadata={})
    skipped = CellResult.skipped(task="t", config="astrojax-multigpu", dtype="f32",
                                 batch_size=1000, reason=SkipReason.BELOW_MULTIGPU_MIN_BATCH)
    cells = [base, candidate, skipped]
    compute_speedup_vs_baseline(cells, baseline_config="brahe-rust-rayon")
    assert base.speedup_vs_baseline == 1.0
    assert candidate.speedup_vs_baseline == pytest.approx(10.0)
    assert skipped.speedup_vs_baseline is None


def test_benchmark_run_save_and_load(tmp_path: Path):
    run = BenchmarkRun(
        run_id="test-run",
        started_at="2026-05-22T19:00:00Z",
        finished_at="2026-05-22T19:01:00Z",
        seed=42,
        iterations=10,
        scheduling=SchedulingPolicy(per_cell_budget_s=120.0, global_run_budget_s=3600.0),
        system=_make_system(),
        data_alignment={"eop_file": "x", "eop_sha256": "y"},
        cells=[
            CellResult.ok_cell(task="t", config="brahe-rust-rayon", dtype="f64",
                               batch_size=10, times=[0.001] * 10, metadata={}),
        ],
    )
    path = run.save(tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["schema_version"] == "1"
    assert loaded["run_id"] == "test-run"
    assert loaded["cells"][0]["batch_size"] == 10
