"""
Accuracy harness for comparative benchmarks.

For each task this module:

1. Builds a single ``params`` dict containing N independent inputs (via
   :meth:`BenchmarkTask.generate_accuracy_samples`).
2. Runs every participating language once with ``iterations=1`` so each
   language emits N results.
3. Applies :meth:`BenchmarkTask.post_process` per language to align frames,
   units, and conventions to the OreKit baseline.
4. Computes per-sample max-abs / rms errors against OreKit and aggregates
   to p50 / p95 / p99 / max distributional statistics.

Performance timing lives in the existing ``runner.py`` ``run`` command; the
two harnesses share task definitions but write to separate output files.

Output: ``benchmarks/comparative/results/accuracy_<timestamp>.jsonl`` (and
the canonical ``accuracy_latest.jsonl``). One JSON object per line, with a
``"kind"`` discriminator of either ``"summary"`` (one per task/comparison)
or ``"sample"`` (one per task/comparison/sample). See
:class:`results.AccuracySummary` and :class:`results.AccuracySample`.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TextIO

import typer

from benchmarks.comparative.config import (
    DEFAULT_SEED,
    NYX_BINARY,
    RESULTS_DIR,
    RUST_BINARY,
    collect_system_info,
)
from benchmarks.comparative.registry import filter_tasks
from benchmarks.comparative.reporting import console
from benchmarks.comparative.results import (
    AccuracySample,
    AccuracySummary,
)
from benchmarks.comparative.tasks.base import BenchmarkTask


def _append_jsonl(stream: TextIO, record: dict) -> None:
    """Append one compact JSON record to an open stream and flush.

    Flushing after every record means a crash mid-run still leaves a
    parseable JSONL file with everything written so far — the previous
    "build the full list then write once" approach lost an entire 9-module
    sweep when a single task hung. Cost is one extra syscall per record;
    the per-iteration cost is negligible next to the work each task does.
    """
    stream.write(json.dumps(record, default=str, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


# Languages other than the baseline are compared against OreKit. The order
# determines the order of comparison records in the JSONL file, which is
# also the order plots will iterate over.
BASELINE_LANGUAGE = "java"
COMPARISON_LANGUAGES = ["gmat", "basilisk", "nyx", "python", "rust"]


def run_accuracy(
    module: Optional[str] = None,
    task: Optional[str] = None,
    samples: int = 100,
    seed: int = DEFAULT_SEED,
    output: Optional[Path] = None,
    quick: bool = False,
) -> Path:
    """Run the accuracy harness across selected tasks.

    Returns the path of the written JSONL file (also written to
    ``accuracy_latest.jsonl``).

    ``quick=True`` overrides every task's sample count to 5 for smoke
    testing the full pipeline without committing to a full run.
    """
    tasks = filter_tasks(module=module, task_name=task)
    if not tasks:
        console.print("[red]No matching tasks found.[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold]Accuracy sweep: {len(tasks)} task(s), "
        f"seed={seed}, default N={samples}{' (quick mode)' if quick else ''}[/bold]\n"
    )

    # No Basilisk/GMAT pre-import dance needed: GMAT runs in a subprocess
    # per task (see _dispatch_one), so process isolation prevents the
    # Spacecraft-type registration conflict that the in-process path
    # had to work around.

    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = output or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_for_filename = timestamp.replace(":", "-").replace("+00-00", "Z").split(".")[0]
    archival_path = output_dir / f"accuracy_{ts_for_filename}.jsonl"
    latest_path = output_dir / "accuracy_latest.jsonl"

    # Open both files for incremental append. Writing to two handles means
    # a crash leaves both files self-consistent — the archival file gets
    # whatever was reached, and accuracy_latest.jsonl always points at the
    # newest progress without a post-run copy step.
    archival_handle = open(archival_path, "w")
    latest_handle = open(latest_path, "w")

    def emit(record: dict) -> None:
        _append_jsonl(archival_handle, record)
        _append_jsonl(latest_handle, record)

    emit(
        {
            "kind": "run_metadata",
            "timestamp": timestamp,
            "seed": seed,
            "default_samples": samples,
            "quick": quick,
            "baseline": BASELINE_LANGUAGE,
            "system_info": collect_system_info(),
        }
    )

    for t in tasks:
        n = 5 if quick else min(samples, t.default_accuracy_samples)
        console.print(f"[cyan]Task:[/cyan] {t.name} (N={n})")

        params = t.generate_accuracy_samples(seed, n)

        # Run every participating language. iterations=1 because for
        # accuracy we want the result, not a timing average.
        per_language_results: dict[str, list] = {}
        per_language_failed: dict[str, str] = {}
        for lang in [BASELINE_LANGUAGE] + COMPARISON_LANGUAGES:
            if lang not in t.languages:
                continue
            console.print(f"  [dim]Running {lang}...[/dim]", end=" ")
            result = _dispatch_one(t, lang, params)
            if result is None:
                per_language_failed[lang] = "dispatch failed"
                console.print("[red]FAILED[/red]")
                continue
            try:
                aligned = t.post_process(lang, result)
            except Exception as e:
                per_language_failed[lang] = f"post_process: {e}"
                console.print(f"[red]post_process error: {e}[/red]")
                continue
            per_language_results[lang] = aligned
            console.print("[green]ok[/green]")

        if BASELINE_LANGUAGE not in per_language_results:
            console.print(
                f"  [yellow]Skipping comparisons: {BASELINE_LANGUAGE} baseline "
                f"unavailable for {t.name}[/yellow]"
            )
            console.print()
            continue

        baseline_results = per_language_results[BASELINE_LANGUAGE]
        for lang in COMPARISON_LANGUAGES:
            if lang not in per_language_results:
                continue
            comp_results = per_language_results[lang]
            samples_for_pair, summary = _compare_samples(
                t,
                baseline_results,
                comp_results,
                lang,
                params,
            )
            emit(summary.to_jsonl_dict())
            for s in samples_for_pair:
                emit(s.to_jsonl_dict())

        console.print()

    archival_handle.close()
    latest_handle.close()
    console.print(f"[dim]Accuracy results written to {archival_path}[/dim]")

    return archival_path


def _dispatch_one(task: BenchmarkTask, language: str, params: dict):
    """Run one (task, language) accuracy invocation. Returns the raw result
    list as emitted by the language backend (no post-processing applied
    here — :meth:`post_process` is the caller's responsibility).
    """
    input_data = {
        "task": task.name,
        "iterations": 1,
        "params": params,
    }
    if language == "python":
        from benchmarks.comparative.implementations.python import dispatch

        try:
            tr = dispatch(input_data)
            return tr.results if tr else None
        except Exception as e:
            console.print(f"    [red]python error: {e}[/red]")
            return None
    if language == "basilisk":
        try:
            from benchmarks.comparative.implementations.basilisk import dispatch
        except ImportError:
            return None
        try:
            tr = dispatch(input_data)
            return tr.results if tr else None
        except Exception as e:
            console.print(f"    [red]basilisk error: {e}[/red]")
            return None
    if language == "gmat":
        # Subprocess-dispatch GMAT so each task starts with a fresh GMAT
        # library state — running GMAT in-process across many tasks
        # accumulates C++ state that segfaults on
        # ``force_model.accel_point_mass_gravity`` (Pure virtual function
        # called!) after a long sweep. A subprocess per task is sequential
        # (no parallelism) so each task still has the whole machine's
        # cores for its work.
        import os
        import sys

        if not os.environ.get("GMAT_ROOT_PATH"):
            return None
        cmd = [sys.executable, "-m", "benchmarks.comparative.implementations.gmat"]
        return _run_subprocess(task, "gmat", input_data, cmd)
    if language == "nyx":
        cmd = [str(NYX_BINARY)] if NYX_BINARY.exists() else None
        return _run_subprocess(task, "nyx", input_data, cmd)
    if language == "rust":
        cmd = [str(RUST_BINARY)] if RUST_BINARY.exists() else None
        return _run_subprocess(task, "rust", input_data, cmd)
    if language == "java":
        from benchmarks.comparative.runner import _get_java_command, _ensure_java_home

        _ensure_java_home()
        cmd = _get_java_command()
        return _run_subprocess(task, "java", input_data, cmd)
    return None


def _run_subprocess(
    task: BenchmarkTask, language: str, input_data: dict, command: Optional[list[str]]
):
    if command is None:
        return None
    try:
        proc = subprocess.run(
            command,
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=task.timeout,
        )
        if proc.returncode == 2 and language == "gmat":
            # Exit 2 from the GMAT subprocess means "not configured" —
            # render the same skip-style message the legacy in-process
            # path produced when GMAT_ROOT_PATH was missing.
            return None
        if proc.returncode != 0:
            console.print(f"    [red]{language} subprocess: {proc.stderr[:200]}[/red]")
            return None
        output = json.loads(proc.stdout)
        return output["results"]
    except subprocess.TimeoutExpired:
        console.print(f"    [red]{language} timeout[/red]")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        console.print(f"    [red]{language} protocol: {e}[/red]")
        return None


def _compare_samples(
    task: BenchmarkTask,
    baseline_results: list,
    comp_results: list,
    comp_language: str,
    params: dict,
) -> tuple[list[AccuracySample], AccuracySummary]:
    """Build per-sample :class:`AccuracySample` records and an aggregated
    :class:`AccuracySummary`.

    Uses ``task.compare_results`` on each pair of single-sample results so
    task-specific comparators (e.g. Euler-angle wrapping, quaternion
    rotation-matrix Frobenius norm) apply. Each per-sample comparison is
    obtained by calling ``compare_results`` with single-element lists.
    """
    n = min(len(baseline_results), len(comp_results))
    sample_keys = _sample_keys_from_params(task, params, n)

    sample_records: list[AccuracySample] = []
    max_abs_values: list[float] = []
    rms_values: list[float] = []
    n_failed = 0

    for i in range(n):
        try:
            cmp = task.compare_results(
                [baseline_results[i]],
                [comp_results[i]],
                BASELINE_LANGUAGE,
                comp_language,
            )
        except Exception as e:
            console.print(f"    [yellow]sample {i} compare failed: {e}[/yellow]")
            n_failed += 1
            continue

        if math.isnan(cmp.max_abs_error) or math.isnan(cmp.rms_error):
            n_failed += 1
            continue

        max_abs_values.append(cmp.max_abs_error)
        rms_values.append(cmp.rms_error)

        # Per-sample richer metrics (computed from the actual result
        # values, not from params). Default is {}; tasks like access
        # override to surface, e.g. contact-count and per-window
        # start/end timing residuals so a task-specific CSV writer can
        # break out the comparison into multiple metric columns.
        try:
            extra_metrics = task.detailed_sample_metrics(
                baseline_results[i], comp_results[i]
            )
        except Exception as e:
            console.print(
                f"    [yellow]sample {i} detailed_sample_metrics failed: {e}[/yellow]"
            )
            extra_metrics = {}

        merged_key = dict(sample_keys[i] if i < len(sample_keys) else {})
        merged_key.update(extra_metrics)

        sample_records.append(
            AccuracySample(
                task_name=task.name,
                reference_language=BASELINE_LANGUAGE,
                comparison_language=comp_language,
                sample_index=i,
                max_abs_error=cmp.max_abs_error,
                rms_error=cmp.rms_error,
                sample_key=merged_key,
            )
        )

    summary = AccuracySummary(
        task_name=task.name,
        reference_language=BASELINE_LANGUAGE,
        comparison_language=comp_language,
        n_samples=len(max_abs_values),
        n_failed=n_failed,
        max_abs_p50=_percentile(max_abs_values, 50),
        max_abs_p95=_percentile(max_abs_values, 95),
        max_abs_p99=_percentile(max_abs_values, 99),
        max_abs_max=max(max_abs_values) if max_abs_values else float("nan"),
        rms_p50=_percentile(rms_values, 50),
        rms_p95=_percentile(rms_values, 95),
        rms_p99=_percentile(rms_values, 99),
        rms_max=max(rms_values) if rms_values else float("nan"),
    )
    return sample_records, summary


def _sample_keys_from_params(task: BenchmarkTask, params: dict, n: int) -> list[dict]:
    """Best-effort extraction of one ``sample_key`` per sample from the
    batched ``params`` dict.

    For tasks that put N inputs into a list under one key (e.g.
    ``params["quaternions"]`` or ``params["cases"]``), iterate the list
    and ask the task to summarize each element. Tasks with no obvious
    list-of-inputs structure return empty dicts, which is fine — the
    plotting code skips scatter-plot generation when sample_key is empty.
    """
    # Common pattern: params is a single-key dict where the value is a
    # list of N inputs. Find it heuristically.
    list_value: list | None = None
    list_key: str | None = None
    for k, v in params.items():
        if isinstance(v, list) and len(v) == n:
            list_value = v
            list_key = k
            break

    if list_value is None or list_key is None:
        # No obvious per-sample structure; return empty keys.
        return [{} for _ in range(n)]

    keys: list[dict] = []
    for elem in list_value:
        sub_params = {list_key: elem} if not isinstance(elem, dict) else elem
        try:
            keys.append(task.accuracy_sample_key(sub_params))
        except Exception:
            keys.append({})
    return keys


def _percentile(values: list[float], pct: float) -> float:
    """Compute the ``pct``-th percentile via linear interpolation.

    Returns NaN for an empty input. Hand-rolled rather than importing
    statistics.quantiles to keep dependencies minimal and to make the
    semantics explicit (linear, inclusive-bracket).
    """
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (pct / 100) * (len(sorted_vals) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_vals[lower]
    fraction = pos - lower
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * fraction
