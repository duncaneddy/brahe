"""
Comparative benchmark orchestrator CLI.

Usage:
    python -m benchmarks.comparative.runner list
    python -m benchmarks.comparative.runner run [--module coordinates] [--language python] [--iterations 100]
    python -m benchmarks.comparative.runner plot
"""

import json
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Optional

import typer

from benchmarks.comparative.config import (
    DEFAULT_ITERATIONS,
    DEFAULT_SEED,
    JAVA_PROJECT_DIR,
    RESULTS_DIR,
    RUST_BINARY,
    collect_system_info,
)
from benchmarks.comparative.registry import filter_tasks
from benchmarks.comparative.reporting import (
    console,
    print_accuracy_table,
    print_performance_table,
    print_task_list,
)
from benchmarks.comparative.results import BenchmarkRun, TaskResult
from benchmarks.comparative.tasks.base import BenchmarkTask

app = typer.Typer(help="Comparative benchmark framework for Brahe")

# Java/OreKit is the reference baseline — run it first, use it for speedup and accuracy
LANGUAGE_ORDER = ["java", "python", "rust"]
BASELINE_LANGUAGE = "java"


@app.command("list")
def list_tasks(
    module: Optional[str] = typer.Option(None, help="Filter by module"),
    language: Optional[str] = typer.Option(None, help="Filter by language support"),
):
    """List available benchmark tasks."""
    tasks = filter_tasks(module=module, language=language)
    print_task_list(tasks)


@app.command()
def run(
    module: Optional[str] = typer.Option(None, help="Filter by module"),
    task: Optional[str] = typer.Option(None, help="Run specific task by name"),
    language: Optional[str] = typer.Option(None, help="Run only this language"),
    iterations: int = typer.Option(DEFAULT_ITERATIONS, help="Number of iterations"),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed for parameter generation"),
    output: Optional[Path] = typer.Option(None, help="Output directory for results"),
):
    """Run comparative benchmarks."""
    tasks = filter_tasks(module=module, task_name=task)
    if not tasks:
        console.print("[red]No matching tasks found.[/red]")
        raise typer.Exit(1)

    # Determine which languages to run
    languages_to_run = [language] if language else None

    console.print(
        f"[bold]Running {len(tasks)} task(s), {iterations} iterations, seed={seed}[/bold]\n"
    )

    benchmark_run = BenchmarkRun(system_info=collect_system_info())

    for t in tasks:
        requested = languages_to_run or t.languages
        # Sort by LANGUAGE_ORDER to ensure baseline (java) runs first
        task_languages = sorted(
            requested,
            key=lambda lang: (
                LANGUAGE_ORDER.index(lang) if lang in LANGUAGE_ORDER else 99
            ),
        )
        console.print(f"[cyan]Task:[/cyan] {t.name} — {t.description}")

        task_results: dict[str, TaskResult] = {}

        for lang in task_languages:
            if lang not in t.languages:
                console.print(f"  [yellow]Skipping {lang} (no implementation)[/yellow]")
                continue

            console.print(f"  [dim]Running {lang}...[/dim]", end=" ")
            result = _dispatch_task(t, lang, iterations, seed)

            if result:
                task_results[lang] = result
                benchmark_run.task_results.append(result)
                console.print(f"[green]mean={result.mean:.6f}s[/green]")
            else:
                console.print("[red]FAILED[/red]")

        # Compute accuracy comparisons with java as reference when available
        if BASELINE_LANGUAGE in task_results:
            for lang in task_results:
                if lang != BASELINE_LANGUAGE:
                    comparison = t.compare_results(
                        task_results[BASELINE_LANGUAGE].results,
                        task_results[lang].results,
                        BASELINE_LANGUAGE,
                        lang,
                    )
                    benchmark_run.accuracy_comparisons.append(comparison)
        else:
            # No baseline; compare all pairs
            lang_pairs = list(combinations(task_results.keys(), 2))
            for lang_a, lang_b in lang_pairs:
                comparison = t.compare_results(
                    task_results[lang_a].results,
                    task_results[lang_b].results,
                    lang_a,
                    lang_b,
                )
                benchmark_run.accuracy_comparisons.append(comparison)

        console.print()

    # Print results
    print_performance_table(benchmark_run)
    print_accuracy_table(benchmark_run)

    # Save results
    output_dir = output or RESULTS_DIR
    filepath = benchmark_run.save(output_dir)
    console.print(f"\n[dim]Results saved to {filepath}[/dim]")


@app.command()
def plot(
    results_file: Optional[Path] = typer.Option(
        None, help="Specific results file to plot"
    ),
):
    """Generate comparison charts from benchmark results."""
    from benchmarks.comparative.plotting import generate_all_plots

    if results_file:
        benchmark_run = BenchmarkRun.load(results_file)
    else:
        benchmark_run = BenchmarkRun.load_latest(RESULTS_DIR)

    if benchmark_run is None:
        console.print("[red]No benchmark results found. Run benchmarks first.[/red]")
        raise typer.Exit(1)

    paths = generate_all_plots(benchmark_run)
    for p in paths:
        console.print(f"[green]Generated:[/green] {p}")


def _dispatch_task(
    task: BenchmarkTask,
    language: str,
    iterations: int,
    seed: int,
) -> TaskResult | None:
    """Dispatch a task to the appropriate language implementation."""
    if language == "python":
        return _run_python(task, iterations, seed)
    elif language == "rust":
        return _run_subprocess(task, language, iterations, seed, _get_rust_command())
    elif language == "java":
        return _run_subprocess(task, language, iterations, seed, _get_java_command())
    return None


def _run_python(
    task: BenchmarkTask,
    iterations: int,
    seed: int,
) -> TaskResult | None:
    """Run a benchmark task using the Python brahe implementation."""
    from benchmarks.comparative.implementations.python import dispatch

    try:
        input_data = task.to_input_json(iterations, seed)
        return dispatch(input_data)
    except Exception as e:
        console.print(f"    [red]Error: {e}[/red]")
        return None


def _get_rust_command() -> list[str] | None:
    """Get the Rust benchmark binary command, or None if not built."""
    if RUST_BINARY.exists():
        return [str(RUST_BINARY)]
    return None


def _get_java_command() -> list[str] | None:
    """Get the Java benchmark command, or None if not built."""
    gradlew = JAVA_PROJECT_DIR / "gradlew"
    if not gradlew.exists():
        return None

    build_dir = JAVA_PROJECT_DIR / "build"
    if not build_dir.exists():
        return None

    return [
        str(gradlew),
        "-p",
        str(JAVA_PROJECT_DIR),
        "--quiet",
        "run",
    ]


def _run_subprocess(
    task: BenchmarkTask,
    language: str,
    iterations: int,
    seed: int,
    command: list[str] | None,
) -> TaskResult | None:
    """Run a benchmark task via subprocess with JSON protocol."""
    if command is None:
        console.print(
            f"    [yellow]{language} not ready. Run: just bench-compare-setup[/yellow]"
        )
        return None

    input_data = task.to_input_json(iterations, seed)
    input_json = json.dumps(input_data)

    try:
        result = subprocess.run(
            command,
            input=input_json,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            console.print(f"    [red]Subprocess error: {result.stderr[:200]}[/red]")
            return None

        output = json.loads(result.stdout)
        return TaskResult(
            task_name=output["task"],
            language=output["metadata"]["language"],
            library=output["metadata"]["library"],
            iterations=output["iterations"],
            times_seconds=output["times_seconds"],
            results=output["results"],
            metadata=output["metadata"],
        )
    except subprocess.TimeoutExpired:
        console.print("    [red]Timeout after 300s[/red]")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        console.print(f"    [red]Protocol error: {e}[/red]")
        return None


def main():
    app()


if __name__ == "__main__":
    main()
