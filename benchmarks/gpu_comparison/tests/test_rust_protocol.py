"""End-to-end check of the Rust subprocess JSON protocol.

Skipped if the `bench_gpu_rust` binary hasn't been built."""
import pytest

from benchmarks.gpu_comparison.implementations.rust_backend import (
    RUST_BINARY,
    run_rust_cell,
)
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


class _PingTask(BatchTask):
    name = "ping.identity"
    module = "ping"
    description = "echo task for protocol validation"
    configs = [BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust")]

    def batch_sizes(self): return [1]
    def generate_inputs(self, b, s): return {"n": b}


@pytest.mark.skipif(not RUST_BINARY.exists(),
                    reason="bench_gpu_rust binary not built")
def test_ping_cell_returns_ok():
    cfg = BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust")
    cell = run_rust_cell(task=_PingTask(), config=cfg, batch_size=1,
                        iterations=3, seed=42, per_cell_budget_s=30.0)
    assert cell.status == "ok"
    assert cell.iterations == 3
    assert cell.throughput_ops_per_sec > 0
    assert cell.metadata["backend_extra"]["task"] == "ping.identity"


@pytest.mark.skipif(not RUST_BINARY.exists(),
                    reason="bench_gpu_rust binary not built")
def test_unknown_task_returns_backend_error():
    class _Unknown(BatchTask):
        name = "ping.unknown"
        module = "ping"
        description = ""
        configs = [BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust")]
        def batch_sizes(self): return [1]
        def generate_inputs(self, b, s): return {}

    cfg = BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust")
    cell = run_rust_cell(task=_Unknown(), config=cfg, batch_size=1,
                        iterations=1, seed=42, per_cell_budget_s=30.0)
    assert cell.status == "skipped"
    assert cell.skip_reason == "backend_error"
