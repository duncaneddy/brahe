"""
Base class for benchmark task specifications.
"""

import math
from abc import ABC, abstractmethod

from benchmarks.comparative.results import AccuracyComparison


class BenchmarkTask(ABC):
    """Abstract base class for benchmark task definitions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Dotted task name, e.g. 'coordinates.geodetic_to_ecef'."""
        ...

    @property
    @abstractmethod
    def module(self) -> str:
        """Module grouping, e.g. 'coordinates'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    @abstractmethod
    def languages(self) -> list[str]:
        """Languages that have implementations."""
        ...

    @property
    def timeout(self) -> int:
        """Subprocess timeout in seconds. Override for slow tasks."""
        return 300

    @abstractmethod
    def generate_params(self, seed: int) -> dict:
        """Generate deterministic benchmark parameters from seed."""
        ...

    def compare_results(
        self,
        results_a: list,
        results_b: list,
        language_a: str,
        language_b: str,
    ) -> AccuracyComparison:
        """Compare numerical results between two implementations.

        Default implementation flattens nested lists and computes element-wise errors.
        """
        flat_a = _flatten(results_a)
        flat_b = _flatten(results_b)

        if not flat_a or not flat_b:
            return AccuracyComparison(
                task_name=self.name,
                reference_language=language_a,
                comparison_language=language_b,
                max_abs_error=float("nan"),
                max_rel_error=float("nan"),
                rms_error=float("nan"),
            )

        n = min(len(flat_a), len(flat_b))
        abs_errors = [abs(flat_a[i] - flat_b[i]) for i in range(n)]
        rel_errors = [
            abs(flat_a[i] - flat_b[i]) / max(abs(flat_a[i]), 1e-30) for i in range(n)
        ]
        rms = math.sqrt(sum(e * e for e in abs_errors) / n)

        return AccuracyComparison(
            task_name=self.name,
            reference_language=language_a,
            comparison_language=language_b,
            max_abs_error=max(abs_errors),
            max_rel_error=max(rel_errors),
            rms_error=rms,
        )

    def to_input_json(self, iterations: int, seed: int) -> dict:
        """Build the JSON input dict for subprocess dispatch."""
        return {
            "task": self.name,
            "iterations": iterations,
            "params": self.generate_params(seed),
        }


def _flatten(lst: list) -> list[float]:
    """Recursively flatten nested lists into a flat list of floats."""
    out = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        elif isinstance(item, (int, float)):
            out.append(float(item))
    return out
