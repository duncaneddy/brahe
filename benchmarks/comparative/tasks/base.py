"""
Base class for benchmark task specifications.
"""

import math
from abc import ABC, abstractmethod
from typing import Any

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

    @property
    def default_accuracy_samples(self) -> int:
        """Default sample count for the accuracy harness. Override on tasks
        whose per-sample cost makes 100 impractical (e.g. propagation with
        80x80 + drag + SRP)."""
        return 100

    @abstractmethod
    def generate_params(self, seed: int) -> dict:
        """Generate deterministic benchmark parameters from seed."""
        ...

    def generate_accuracy_samples(self, seed: int, n: int) -> dict:
        """Build a params dict containing `n` independent inputs for the
        accuracy harness.

        Default: delegates to `generate_params(seed)`, which preserves
        today's behavior for tasks whose natural batch is already the right
        coverage (e.g. attitude tasks with 50 quaternions). Tasks whose
        helper takes an explicit count should override to thread `n`
        through.

        Tasks where one big input is the whole point (e.g. the 48-hour
        access task) may simply return `generate_params(seed)` and let the
        harness's "treat the single output as N=1" path take over.
        """
        return self.generate_params(seed)

    def post_process(self, language: str, result: Any) -> Any:
        """Apply per-language alignment (frames, units, conventions) to a
        result before accuracy comparison.

        Default: identity. Tasks override when a language reports values in
        a frame, unit, or convention that differs from the OreKit baseline.

        This is the single place to declare those alignments so the
        comparison logic lives next to the task spec rather than inside the
        language-specific runner modules.
        """
        return result

    def accuracy_sample_key(self, params: dict) -> dict:
        """Return a dict of scalars describing each sample, used as the
        x-axis for accuracy scatter plots and to surface in JSONL detail
        records.

        Default: empty dict (only CDF plots are emitted). Tasks where one
        parameter dominates (altitude for orbit-based tasks, epoch for
        time-based tasks) override to return e.g. `{"altitude_km": ...}`.
        """
        return {}

    def detailed_sample_metrics(
        self,
        baseline_sample,
        comparison_sample,
    ) -> dict:
        """Return per-sample metrics computed from the actual result
        values (not from params). Merged into ``AccuracySample.sample_key``
        by the accuracy harness so a task-specific CSV writer can break
        the comparison out into multiple metric columns / rows.

        Default: empty dict — tasks whose ``compare_results`` summary
        captures everything they care about don't need to populate this.
        Access overrides this to surface contact-count and per-window
        start/end timing residuals alongside the rolled-up max_abs.
        """
        return {}

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
