import pytest

from benchmarks.gpu_comparison.registry import filter_tasks, list_tasks, register
from benchmarks.gpu_comparison.tasks.base import BatchConfig, BatchTask


class _CoordTask(BatchTask):
    name = "coordinates.foo"
    module = "coordinates"
    description = "stub"
    configs = [BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust")]

    def batch_sizes(self): return [1]
    def generate_inputs(self, b, s): return {}


class _TimeTask(BatchTask):
    name = "time.bar"
    module = "time"
    description = "stub"
    configs = [BatchConfig(name="brahe-rust-rayon", dtype="f64", backend="rust")]

    def batch_sizes(self): return [1]
    def generate_inputs(self, b, s): return {}


@pytest.fixture(autouse=True)
def _clear_registry(monkeypatch):
    from benchmarks.gpu_comparison import registry
    monkeypatch.setattr(registry, "_REGISTRY", {})


def test_register_and_list():
    register(_CoordTask)
    register(_TimeTask)
    names = [t.name for t in list_tasks()]
    assert "coordinates.foo" in names
    assert "time.bar" in names


def test_filter_by_module():
    register(_CoordTask)
    register(_TimeTask)
    tasks = filter_tasks(module="coordinates")
    assert [t.name for t in tasks] == ["coordinates.foo"]


def test_filter_by_task_name():
    register(_CoordTask)
    register(_TimeTask)
    tasks = filter_tasks(task_name="time.bar")
    assert [t.name for t in tasks] == ["time.bar"]


def test_filter_by_backend():
    class _GpuOnly(BatchTask):
        name = "x.y"
        module = "x"
        description = "stub"
        configs = [BatchConfig(name="astrojax-gpu", dtype="f32", backend="astrojax-gpu")]

        def batch_sizes(self): return [1]
        def generate_inputs(self, b, s): return {}

    register(_GpuOnly)
    register(_CoordTask)
    rust_supporting = filter_tasks(backend="rust")
    assert [t.name for t in rust_supporting] == ["coordinates.foo"]


def test_double_register_raises():
    register(_CoordTask)
    with pytest.raises(ValueError):
        register(_CoordTask)
