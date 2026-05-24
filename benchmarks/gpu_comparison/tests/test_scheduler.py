from dataclasses import dataclass

import pytest

from benchmarks.gpu_comparison.results import (
    CellResult,
    SchedulingPolicy,
    SkipReason,
)
from benchmarks.gpu_comparison.scheduler import (
    project_next_cell_time,
    should_run_multigpu,
    schedule_ladder,
)
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


class _StubTask(BatchTask):
    name = "stub.task"
    module = "stub"
    description = "stub"
    configs = [
        BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust"),
        BatchConfig(name="astrojax-multigpu", dtype="f32", backend="astrojax-multigpu"),
    ]

    def batch_sizes(self): return [10, 100, 1000, 10_000]
    def generate_inputs(self, b, s): return {"n": b}
    def multigpu_min_batch(self) -> int: return 500


def test_should_run_multigpu():
    t = _StubTask()
    assert not should_run_multigpu(t, 10)
    assert not should_run_multigpu(t, 100)
    assert should_run_multigpu(t, 500)
    assert should_run_multigpu(t, 10_000)


def test_project_next_cell_time_scales_linearly():
    prev = CellResult.ok_cell(task="t", config="c", dtype="f64",
                              batch_size=100, times=[0.001] * 10, metadata={})
    proj = project_next_cell_time(prev_cell=prev, prev_batch=100,
                                  next_batch=1000, iterations=10)
    # per-iter at next batch = 0.001 * (1000/100) = 0.01; total = 10 * 0.01 = 0.1
    assert proj == pytest.approx(0.1)


def test_schedule_ladder_runs_within_budget():
    t = _StubTask()
    config = t.configs[0]

    def fake_run(batch_size: int) -> CellResult:
        return CellResult.ok_cell(
            task=t.name, config=config.name, dtype=config.dtype,
            batch_size=batch_size, times=[1e-6] * 10, metadata={},
        )

    policy = SchedulingPolicy(per_cell_budget_s=120.0, global_run_budget_s=3600.0)
    cells = schedule_ladder(task=t, config=config, run_one_cell=fake_run, policy=policy)
    assert len(cells) == 4
    assert all(c.status == "ok" for c in cells)


def test_schedule_ladder_projected_skip():
    t = _StubTask()
    config = t.configs[0]

    def fake_run(batch_size: int) -> CellResult:
        return CellResult.ok_cell(
            task=t.name, config=config.name, dtype=config.dtype,
            batch_size=batch_size, times=[10.0] * 10, metadata={},
        )

    policy = SchedulingPolicy(per_cell_budget_s=120.0, global_run_budget_s=3600.0)
    cells = schedule_ladder(task=t, config=config, run_one_cell=fake_run, policy=policy)
    assert cells[0].status == "ok"
    for c in cells[1:]:
        assert c.status == "skipped"
        assert c.skip_reason == "budget_projected_exceeded"


def test_schedule_ladder_multigpu_skips_small_batches():
    t = _StubTask()
    multigpu_config = t.configs[1]

    def fake_run(batch_size: int) -> CellResult:
        return CellResult.ok_cell(
            task=t.name, config=multigpu_config.name, dtype=multigpu_config.dtype,
            batch_size=batch_size, times=[1e-6] * 10, metadata={},
        )

    policy = SchedulingPolicy(per_cell_budget_s=120.0, global_run_budget_s=3600.0)
    cells = schedule_ladder(task=t, config=multigpu_config, run_one_cell=fake_run, policy=policy)
    assert cells[0].skip_reason == "below_multigpu_min_batch"
    assert cells[1].skip_reason == "below_multigpu_min_batch"
    assert cells[2].status == "ok"
    assert cells[3].status == "ok"
