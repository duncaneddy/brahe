from dataclasses import FrozenInstanceError

import pytest

from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


def test_batch_config_is_frozen():
    cfg = BatchConfig(name="x", dtype="f64", backend="rust")
    with pytest.raises(FrozenInstanceError):
        cfg.name = "y"


def test_batch_config_equality_by_value():
    a = BatchConfig(name="x", dtype="f64", backend="rust")
    b = BatchConfig(name="x", dtype="f64", backend="rust")
    assert a == b
    assert hash(a) == hash(b)


class _StubTask(BatchTask):
    name = "stub.task"
    module = "stub"
    description = "stub"
    configs = [BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust")]

    def batch_sizes(self) -> list[int]:
        return [1, 10, 100]

    def generate_inputs(self, batch_size: int, seed: int) -> dict:
        return {"n": batch_size, "seed": seed}


def test_default_warmup_iterations():
    t = _StubTask()
    assert t.warmup_iterations("rust") == 3
    assert t.warmup_iterations("astrojax-cpu") == 2
    assert t.warmup_iterations("astrojax-gpu") == 2
    assert t.warmup_iterations("astrojax-multigpu") == 2


def test_default_multigpu_min_batch():
    assert _StubTask().multigpu_min_batch() == 100_000


def test_generate_inputs_is_seeded_deterministic():
    t = _StubTask()
    assert t.generate_inputs(10, 42) == t.generate_inputs(10, 42)


def test_subclass_must_implement_abstracts():
    with pytest.raises(TypeError):
        class _Bad(BatchTask):
            pass
        _Bad()
