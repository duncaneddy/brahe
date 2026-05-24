"""Per-task kernel registry for astrojax backends.

Each task in Phase 4 registers a builder callable here. The builder
receives ``(task, batch_size, dtype, seed, devices)`` and returns a
``(kernel, kernel_args)`` pair, where ``kernel`` is something the
dispatcher can call as ``kernel(kernel_args)`` to produce a JAX array
(or a Python value, for the ping test).

The dispatcher wraps ``kernel`` with the appropriate device placement
and timing logic.
"""

from __future__ import annotations

from typing import Callable

from benchmarks.gpu_comparison.tasks.base import BatchTask


# (task, batch_size, dtype, seed, devices) -> (kernel_fn, kernel_args)
Builder = Callable[[BatchTask, int, str, int, list], tuple[Callable, dict]]


_BUILDERS: dict[str, Builder] = {}


def register(task_name: str, builder: Builder) -> None:
    _BUILDERS[task_name] = builder


def get(task_name: str) -> Builder:
    if task_name not in _BUILDERS:
        raise KeyError(f"no astrojax kernel builder registered for '{task_name}'")
    return _BUILDERS[task_name]
