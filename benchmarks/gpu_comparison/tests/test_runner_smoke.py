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


def test_run_suite_enforces_global_budget(monkeypatch, tmp_path):
    """A near-zero global budget skips every remaining task with the
    `global_budget_exceeded` reason, instead of silently running them all."""
    from benchmarks.gpu_comparison import runner
    from benchmarks.gpu_comparison.tasks import register_all  # noqa: F401

    monkeypatch.setattr(runner, "filter_tasks", lambda **_: [_PingTask()])

    # First time.time() call returns t0 (sets suite_start); every subsequent
    # call returns t0 + 1e6, so elapsed is far past the 1-second budget on
    # the first per-task check.
    real_time = runner.time.time
    t0 = real_time()
    calls = {"n": 0}
    def _fake_time():
        calls["n"] += 1
        return t0 if calls["n"] == 1 else t0 + 1e6
    monkeypatch.setattr(runner.time, "time", _fake_time)

    path = runner.run_suite(
        global_run_budget_s=1.0,
        per_cell_budget_s=30.0,
        iterations=2,
        output_dir=tmp_path,
    )
    import json
    cells = json.loads(path.read_text())["cells"]
    assert cells, "expected at least one skipped cell"
    assert all(c["status"] == "skipped" for c in cells)
    assert all(c["skip_reason"] == "global_budget_exceeded" for c in cells)
