"""Runner smoke test using only the ping backends — no JAX, no Rust binary required."""
import pytest

from benchmarks.gpu_comparison.implementations import astrojax_kernels
from benchmarks.gpu_comparison.runner import run_one_task
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


class _PingTask(BatchTask):
    name = "ping.identity"
    module = "ping"
    description = ""
    configs = [BatchConfig(name="astrojax-gpu", dtype="f32", backend="astrojax-gpu")]
    def batch_sizes(self): return [1, 10]
    def generate_inputs(self, b, s): return {"n": b}


def _ping_builder(task, batch_size, dtype, seed, devices):
    def kernel(args): return args["n"] * 2
    return kernel, {"n": batch_size}


@pytest.fixture(autouse=True)
def _register_ping():
    astrojax_kernels.register("ping.identity", _ping_builder)
    yield
    astrojax_kernels._BUILDERS.pop("ping.identity", None)


def test_run_one_task_produces_cells_for_each_batch(monkeypatch):
    # Force the dispatcher to use the CPU sentinel path so no real JAX init happens.
    from benchmarks.gpu_comparison.implementations import astrojax_gpu

    def _fake_resolve(backend, *, force_cpu=False):
        return ["cpu-sentinel"]

    monkeypatch.setattr(astrojax_gpu, "_resolve_devices", _fake_resolve)

    cells = run_one_task(
        _PingTask(), iterations=3, seed=42,
        per_cell_budget_s=30.0, configs_filter=None,
    )
    assert len(cells) == 2
    assert {c.batch_size for c in cells} == {1, 10}
