"""Global registry of BatchTask subclasses.

Each task module (``time_tasks.py``, ``coordinates_tasks.py``, ...) imports
``register`` and decorates / calls it at import time. The CLI imports the
``tasks`` package once, populating the registry, then queries via
``list_tasks`` / ``filter_tasks``.
"""

from __future__ import annotations

from typing import Optional

from benchmarks.gpu_comparison.tasks.base import BatchTask


_REGISTRY: dict[str, BatchTask] = {}


def register(task_cls: type[BatchTask]) -> type[BatchTask]:
    """Instantiate the task class once and add it to the registry."""
    task = task_cls()
    if task.name in _REGISTRY:
        raise ValueError(f"Task '{task.name}' already registered")
    _REGISTRY[task.name] = task
    return task_cls


def list_tasks() -> list[BatchTask]:
    return sorted(_REGISTRY.values(), key=lambda t: t.name)


def filter_tasks(
    *,
    module: Optional[str] = None,
    task_name: Optional[str] = None,
    backend: Optional[str] = None,
) -> list[BatchTask]:
    tasks = list_tasks()
    if module is not None:
        tasks = [t for t in tasks if t.module == module]
    if task_name is not None:
        tasks = [t for t in tasks if t.name == task_name]
    if backend is not None:
        tasks = [t for t in tasks if any(c.backend == backend for c in t.configs)]
    return tasks
