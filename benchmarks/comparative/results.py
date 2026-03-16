"""
Result dataclasses and JSON serialization for comparative benchmarks.
"""

import json
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TaskResult:
    """Result from running a single benchmark task in one language."""

    task_name: str
    language: str
    library: str
    iterations: int
    times_seconds: list[float]
    results: list  # list of output values per iteration (only first stored)
    metadata: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times_seconds)

    @property
    def std(self) -> float:
        return (
            statistics.stdev(self.times_seconds) if len(self.times_seconds) > 1 else 0.0
        )

    @property
    def median(self) -> float:
        return statistics.median(self.times_seconds)

    @property
    def min(self) -> float:
        return min(self.times_seconds)

    @property
    def max(self) -> float:
        return max(self.times_seconds)

    def summary_dict(self) -> dict:
        """Return summary statistics as a dict."""
        return {
            "task_name": self.task_name,
            "language": self.language,
            "library": self.library,
            "iterations": self.iterations,
            "mean_s": self.mean,
            "std_s": self.std,
            "median_s": self.median,
            "min_s": self.min,
            "max_s": self.max,
        }


@dataclass
class AccuracyComparison:
    """Numerical accuracy comparison between two implementations."""

    task_name: str
    reference_language: str
    comparison_language: str
    max_abs_error: float
    max_rel_error: float
    rms_error: float


@dataclass
class BenchmarkRun:
    """Complete benchmark run with all results and comparisons."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    system_info: dict = field(default_factory=dict)
    task_results: list[TaskResult] = field(default_factory=list)
    accuracy_comparisons: list[AccuracyComparison] = field(default_factory=list)

    def save(self, path: Path) -> Path:
        """Save benchmark run to JSON file."""
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"run_{self.run_id}_{self.timestamp[:10]}.json"

        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "task_results": [asdict(r) for r in self.task_results],
            "accuracy_comparisons": [asdict(a) for a in self.accuracy_comparisons],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "BenchmarkRun":
        """Load benchmark run from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        run = cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            system_info=data.get("system_info", {}),
        )
        for r in data.get("task_results", []):
            run.task_results.append(TaskResult(**r))
        for a in data.get("accuracy_comparisons", []):
            run.accuracy_comparisons.append(AccuracyComparison(**a))
        return run

    @classmethod
    def load_latest(cls, results_dir: Path) -> "BenchmarkRun | None":
        """Load the most recent benchmark run from the results directory."""
        json_files = sorted(results_dir.glob("run_*.json"), reverse=True)
        if not json_files:
            return None
        return cls.load(json_files[0])
