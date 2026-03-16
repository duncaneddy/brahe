"""
Rich console tables for benchmark reporting.
"""

from rich.console import Console
from rich.table import Table

from benchmarks.comparative.results import BenchmarkRun, TaskResult

console = Console()

# OreKit (Java) is the reference baseline
BASELINE_LANGUAGE = "java"


def _format_time(seconds: float) -> str:
    """Format time with appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} \u00b5s"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def print_performance_table(run: BenchmarkRun) -> None:
    """Print a performance comparison table with OreKit as baseline."""
    table = Table(title="Performance Comparison", show_lines=True)
    table.add_column("Task", style="bold")
    table.add_column("Language", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Speedup", justify="right")

    # Group results by task
    tasks: dict[str, list[TaskResult]] = {}
    for r in run.task_results:
        tasks.setdefault(r.task_name, []).append(r)

    for task_name, results in sorted(tasks.items()):
        # Use java/orekit as baseline if present, otherwise use slowest
        baseline_result = next(
            (r for r in results if r.language == BASELINE_LANGUAGE), None
        )
        baseline_mean = (
            baseline_result.mean if baseline_result else max(r.mean for r in results)
        )

        for r in sorted(results, key=lambda x: x.mean):
            speedup = baseline_mean / r.mean if r.mean > 0 else float("inf")
            if r.language == BASELINE_LANGUAGE:
                speedup_str = "[dim]baseline[/dim]"
            elif speedup > 1.01:
                speedup_str = f"[green]{speedup:.1f}x[/green]"
            elif speedup < 0.99:
                speedup_str = f"[red]{1.0 / speedup:.1f}x slower[/red]"
            else:
                speedup_str = "[green]~1x[/green]"

            table.add_row(
                task_name,
                f"{r.language} ({r.library})",
                _format_time(r.mean),
                _format_time(r.median),
                _format_time(r.std),
                _format_time(r.min),
                _format_time(r.max),
                speedup_str,
            )

    console.print(table)


def print_accuracy_table(run: BenchmarkRun) -> None:
    """Print a numerical accuracy comparison table."""
    if not run.accuracy_comparisons:
        return

    table = Table(title="Numerical Accuracy (vs OreKit baseline)", show_lines=True)
    table.add_column("Task", style="bold")
    table.add_column("Reference", style="cyan")
    table.add_column("Comparison", style="cyan")
    table.add_column("Max Abs Error", justify="right")
    table.add_column("Max Rel Error", justify="right")
    table.add_column("RMS Error", justify="right")

    for a in sorted(run.accuracy_comparisons, key=lambda x: x.task_name):
        table.add_row(
            a.task_name,
            a.reference_language,
            a.comparison_language,
            f"{a.max_abs_error:.2e}",
            f"{a.max_rel_error:.2e}",
            f"{a.rms_error:.2e}",
        )

    console.print(table)


def print_task_list(tasks: list) -> None:
    """Print available benchmark tasks."""
    table = Table(title="Available Benchmark Tasks")
    table.add_column("Task", style="bold")
    table.add_column("Module", style="cyan")
    table.add_column("Languages", style="green")
    table.add_column("Description")

    for t in sorted(tasks, key=lambda x: x.name):
        table.add_row(
            t.name,
            t.module,
            ", ".join(t.languages),
            t.description,
        )

    console.print(table)
