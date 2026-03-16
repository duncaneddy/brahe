"""
Time system conversion benchmark task specifications.
"""

import math
import random

from benchmarks.comparative.results import AccuracyComparison
from benchmarks.comparative.tasks.base import BenchmarkTask


def _generate_random_datetimes(seed: int, count: int = 50) -> list[dict]:
    """Generate random datetime components for time benchmarks."""
    rng = random.Random(seed)
    datetimes = []
    for _ in range(count):
        year = rng.randint(2000, 2030)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        second = rng.uniform(0.0, 59.999)
        nanosecond = rng.uniform(0.0, 999999999.0)
        datetimes.append(
            {
                "year": year,
                "month": month,
                "day": day,
                "hour": hour,
                "minute": minute,
                "second": second,
                "nanosecond": nanosecond,
            }
        )
    return datetimes


class EpochCreationTask(BenchmarkTask):
    """Benchmark creating Epochs from datetime components and extracting JD."""

    @property
    def name(self) -> str:
        return "time.epoch_creation"

    @property
    def module(self) -> str:
        return "time"

    @property
    def description(self) -> str:
        return "Create Epoch from datetime components and return Julian Date"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        return {"datetimes": _generate_random_datetimes(seed)}


class UtcToTaiTask(BenchmarkTask):
    """Benchmark UTC to TAI time system conversion."""

    @property
    def name(self) -> str:
        return "time.utc_to_tai"

    @property
    def module(self) -> str:
        return "time"

    @property
    def description(self) -> str:
        return "Convert UTC epoch to TAI Julian Date"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        return {"datetimes": _generate_random_datetimes(seed)}


class UtcToTtTask(BenchmarkTask):
    """Benchmark UTC to TT time system conversion."""

    @property
    def name(self) -> str:
        return "time.utc_to_tt"

    @property
    def module(self) -> str:
        return "time"

    @property
    def description(self) -> str:
        return "Convert UTC epoch to TT Julian Date"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        return {"datetimes": _generate_random_datetimes(seed)}


class UtcToGpsTask(BenchmarkTask):
    """Benchmark UTC to GPS time system conversion."""

    @property
    def name(self) -> str:
        return "time.utc_to_gps"

    @property
    def module(self) -> str:
        return "time"

    @property
    def description(self) -> str:
        return "Convert UTC epoch to GPS Julian Date"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        return {"datetimes": _generate_random_datetimes(seed)}


class UtcToUt1Task(BenchmarkTask):
    """Benchmark UTC to UT1 time system conversion."""

    @property
    def name(self) -> str:
        return "time.utc_to_ut1"

    @property
    def module(self) -> str:
        return "time"

    @property
    def description(self) -> str:
        return "Convert UTC epoch to UT1 Julian Date"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        return {"datetimes": _generate_random_datetimes(seed)}

    def compare_results(
        self,
        results_a: list,
        results_b: list,
        language_a: str,
        language_b: str,
    ) -> AccuracyComparison:
        """Compare UT1 results with wider tolerance.

        Brahe uses zero-valued EOP while OreKit uses real IERS data,
        so UT1 values can diverge by up to ~1 second (~1.16e-5 days in JD).
        """
        n = min(len(results_a), len(results_b))
        if n == 0:
            return AccuracyComparison(
                task_name=self.name,
                reference_language=language_a,
                comparison_language=language_b,
                max_abs_error=float("nan"),
                max_rel_error=float("nan"),
                rms_error=float("nan"),
            )

        abs_errors = [abs(results_a[i] - results_b[i]) for i in range(n)]
        rel_errors = [
            abs(results_a[i] - results_b[i]) / max(abs(results_a[i]), 1e-30)
            for i in range(n)
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
