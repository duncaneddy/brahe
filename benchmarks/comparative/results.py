"""
Result dataclasses and JSON serialization for comparative benchmarks.

Two output families:

- Performance: dataclasses :class:`TaskResult` and :class:`BenchmarkRun`
  serialized as one pretty-printed JSON file per run (legacy format).
- Accuracy: dataclasses :class:`AccuracySample` and :class:`AccuracySummary`
  serialized as JSONL (one JSON object per line) with kind discriminators.
  This keeps per-sample detail recoverable without bloating a single JSON
  document, and stays append-friendly so resuming or splitting accuracy
  runs across languages stays straightforward.
"""

import json
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


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
    """Numerical accuracy comparison between two implementations.

    Used by the legacy performance run (a single sample per task). New code
    should prefer :class:`AccuracySummary` / :class:`AccuracySample`, which
    capture distribution information across an initial-condition sweep.
    """

    task_name: str
    reference_language: str
    comparison_language: str
    max_abs_error: float
    max_rel_error: float
    rms_error: float


@dataclass
class AccuracySample:
    """One initial-condition sample's accuracy result against the baseline."""

    task_name: str
    reference_language: str
    comparison_language: str
    sample_index: int
    max_abs_error: float
    rms_error: float
    sample_key: dict = field(default_factory=dict)

    def to_jsonl_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = "sample"
        return d


@dataclass
class AccuracySummary:
    """Distribution summary of accuracy across an IC sweep."""

    task_name: str
    reference_language: str
    comparison_language: str
    n_samples: int
    n_failed: int
    max_abs_p50: float
    max_abs_p95: float
    max_abs_p99: float
    max_abs_max: float
    rms_p50: float
    rms_p95: float
    rms_p99: float
    rms_max: float

    def to_jsonl_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = "summary"
        return d


def write_jsonl(filepath: Path, records: Iterable[dict]) -> None:
    """Write an iterable of records as one JSON object per line.

    Uses compact separators — accuracy JSONL is read by tooling, not by
    humans, so dropping the pretty indent saves disk and parse time.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str, separators=(",", ":")))
            f.write("\n")


def read_jsonl(filepath: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Skips blank lines."""
    records: list[dict] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


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
        """Save benchmark run to JSON file.

        Each run is saved twice:
          - `run_<ISO-8601 timestamp>.json` — archival, immutable
          - `run_latest.json`               — overwritten on every save

        The ISO-8601 timestamp (UTC, ``-`` substituted for ``:``) gives
        filename uniqueness and lexicographic sort-by-recency, replacing
        the previous UUID-nonce + date-only scheme that wasn't sortable
        beyond per-day granularity.
        """
        path.mkdir(parents=True, exist_ok=True)
        # Filename-safe ISO-8601: 2026-05-16T17-38-42Z (no colons, UTC).
        ts_for_filename = (
            self.timestamp.replace(":", "-").replace("+00-00", "Z").split(".")[0]
        )
        filepath = path / f"run_{ts_for_filename}.json"
        latest_path = path / "run_latest.json"

        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "task_results": [asdict(r) for r in self.task_results],
            "accuracy_comparisons": [asdict(a) for a in self.accuracy_comparisons],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        # Overwrite run_latest.json so downstream tooling (plot generators,
        # docs CI) always has a canonical, stable filename to read.
        with open(latest_path, "w") as f:
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
        """Load the most recent benchmark run from the results directory.

        Prefers ``run_latest.json`` (always written by ``save()``); falls
        back to lexicographically-sorted ``run_<timestamp>.json`` files
        when the canonical name is absent. The fallback explicitly filters
        out the legacy UUID-prefixed filenames (``run_<8 hex>_*.json``)
        because they sort *after* digit-prefixed ISO timestamps in ASCII
        and would otherwise mask newer runs.
        """
        latest_path = results_dir / "run_latest.json"
        if latest_path.exists():
            return cls.load(latest_path)
        candidates = [
            p
            for p in results_dir.glob("run_*.json")
            # ISO-8601 timestamps start with a digit; legacy UUID hex starts
            # with [0-9a-f]. Filter to filenames whose timestamp segment
            # begins with a 4-digit year (i.e. "run_YYYY-...").
            if p.stem.startswith("run_") and len(p.stem) > 8 and p.stem[4:8].isdigit()
        ]
        if not candidates:
            return None
        return cls.load(sorted(candidates, reverse=True)[0])
