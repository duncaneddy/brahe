"""Abstract base classes for the GPU-comparison benchmark suite.

A `BatchTask` describes one workload (e.g. "geodetic → ECEF on N points").
A `BatchConfig` describes how to execute it (which backend, which dtype).
The runner combines tasks and configs into cells, sweeps each (task, config)
across the task's batch-size ladder, and emits one `CellResult` per cell.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchConfig:
    """One (backend, dtype, parallelism) tuple a task can be run under.

    The four configs the suite ships with are constructed in
    ``benchmarks.gpu_comparison.implementations`` and passed into each
    task's ``configs`` field. Tasks may declare a narrower list — e.g.,
    a 20x20 force-model task that astrojax cannot run yet declares only
    the ``brahe-rust-rayon`` config so the scheduler skips the missing
    cells with ``config_not_supported_by_task``.
    """

    name: str
    dtype: str  # "f64" or "f32"
    backend: str  # "rust" | "astrojax-cpu" | "astrojax-gpu" | "astrojax-multigpu"


class BatchTask(ABC):
    """Abstract definition of one benchmark workload."""

    name: str  # e.g. "coordinates.geodetic_to_ecef"
    module: str  # e.g. "coordinates"
    description: str
    configs: list[BatchConfig]

    @abstractmethod
    def batch_sizes(self) -> list[int]:
        """Geometric ladder of batch sizes to sweep, ascending."""

    @abstractmethod
    def generate_inputs(self, batch_size: int, seed: int) -> dict:
        """Deterministic, JSON-serialisable inputs of the requested batch size."""

    def warmup_iterations(self, backend: str) -> int:
        """Number of warmup calls before timed iterations begin."""
        return 3 if backend == "rust" else 2

    def multigpu_min_batch(self) -> int:
        """Smallest batch size for which the astrojax-multigpu config runs.

        Below this threshold pmap inter-device communication dominates and
        multi-GPU is slower than single-GPU. The scheduler emits
        ``below_multigpu_min_batch`` for cells under this size.
        """
        return 100_000
