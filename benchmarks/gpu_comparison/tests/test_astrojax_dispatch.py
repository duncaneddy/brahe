"""Dispatch tests for the astrojax in-process and spawned-CPU backends.

Uses a ping kernel (no actual astrojax import) so the test runs without
JAX. Real astrojax-backed tests live in the per-task test files.
"""

import pytest

from benchmarks.gpu_comparison.implementations import astrojax_kernels
from benchmarks.gpu_comparison.implementations.astrojax_gpu import (
    run_astrojax_cell_in_process,
)
from benchmarks.gpu_comparison.implementations.astrojax_cpu import (
    run_astrojax_cell_in_child,
)
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


class _PingTask(BatchTask):
    name = "ping.identity"
    module = "ping"
    description = "echo task"
    configs = [
        BatchConfig(name="astrojax-cpu", dtype="f64", backend="astrojax-cpu"),
        BatchConfig(name="astrojax-gpu", dtype="f32", backend="astrojax-gpu"),
    ]

    def batch_sizes(self):
        return [1, 10]

    def generate_inputs(self, b, s):
        return {"n": b}


def _ping_builder(task, batch_size, dtype, seed, devices):
    def kernel(args):
        return args["n"] * 2

    return kernel, {"n": batch_size}


@pytest.fixture(autouse=True)
def _register_ping():
    astrojax_kernels.register("ping.identity", _ping_builder)
    yield
    astrojax_kernels._BUILDERS.pop("ping.identity", None)


def test_run_in_process_cpu():
    cfg = BatchConfig(name="astrojax-cpu", dtype="f64", backend="astrojax-cpu")
    cell = run_astrojax_cell_in_process(
        task=_PingTask(),
        config=cfg,
        batch_size=10,
        iterations=3,
        seed=42,
        force_cpu=True,
    )
    assert cell.status == "ok"
    assert cell.batch_size == 10
    assert cell.iterations == 3


def test_run_in_child_returns_cell():
    cfg = BatchConfig(name="astrojax-cpu", dtype="f64", backend="astrojax-cpu")
    cell = run_astrojax_cell_in_child(
        task=_PingTask(),
        config=cfg,
        batch_size=10,
        iterations=3,
        seed=42,
    )
    # The child process has its own astrojax_kernels._BUILDERS dict, so it
    # won't see the parent's registered ping builder. We expect this to
    # surface as CONFIG_NOT_SUPPORTED_BY_TASK rather than a hang or crash —
    # which still verifies the spawn-child plumbing works end-to-end.
    assert cell.status == "skipped"
    assert cell.skip_reason == "config_not_supported_by_task"
