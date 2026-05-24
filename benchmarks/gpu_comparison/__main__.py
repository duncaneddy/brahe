"""CLI: ``python -m benchmarks.gpu_comparison <command>``.

Subcommands:
  list                 — print known tasks, modules, configs.
  run                  — sweep the whole suite; writes one JSON per run.
  run-cell             — execute a single (task, config, batch) cell.
  inspect <path>       — pretty-print a results file as a table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from benchmarks.gpu_comparison.config import RESULTS_DIR
from benchmarks.gpu_comparison.registry import filter_tasks
from benchmarks.gpu_comparison.runner import run_one_cell, run_suite


app = typer.Typer(add_completion=False, help="brahe vs astrojax GPU benchmark suite")
console = Console()


@app.command("list")
def cmd_list(
    module: Optional[str] = typer.Option(None, help="Filter by module"),
):
    import benchmarks.gpu_comparison.tasks.register_all  # noqa: F401 — populates registry
    table = Table(title="Registered tasks")
    table.add_column("name")
    table.add_column("module")
    table.add_column("configs")
    table.add_column("batch sizes")
    for t in filter_tasks(module=module):
        table.add_row(
            t.name, t.module,
            ", ".join(c.name for c in t.configs),
            ", ".join(str(b) for b in t.batch_sizes()),
        )
    console.print(table)


@app.command()
def run(
    module: Optional[str] = typer.Option(None),
    task: Optional[str] = typer.Option(None, help="Run a single task by name"),
    config: Optional[list[str]] = typer.Option(
        None, help="Restrict to these configs (repeatable)",
    ),
    iterations: int = typer.Option(10),
    seed: int = typer.Option(42),
    per_cell_budget_s: float = typer.Option(120.0, "--budget", help="Per-cell wall-clock budget (s)"),
    global_run_budget_s: float = typer.Option(3600.0, "--global-budget"),
    output: Path = typer.Option(RESULTS_DIR, help="Output directory"),
):
    """Run the full suite (filtered by module / task / config)."""
    path = run_suite(
        module=module, task_name=task, configs_filter=config,
        iterations=iterations, seed=seed,
        per_cell_budget_s=per_cell_budget_s,
        global_run_budget_s=global_run_budget_s,
        output_dir=output,
    )
    console.print(f"[green]Results saved to[/green] {path}")


@app.command("run-cell")
def cmd_run_cell(
    task: str = typer.Argument(...),
    config: str = typer.Argument(...),
    batch: int = typer.Argument(...),
    iterations: int = typer.Option(10),
    seed: int = typer.Option(42),
    per_cell_budget_s: float = typer.Option(120.0, "--budget"),
):
    """Run a single (task, config, batch_size) cell. For triage / CI smoke tests."""
    import benchmarks.gpu_comparison.tasks.register_all  # noqa: F401
    matches = filter_tasks(task_name=task)
    if not matches:
        console.print(f"[red]Unknown task: {task}[/red]")
        raise typer.Exit(1)
    t = matches[0]
    cfgs = [c for c in t.configs if c.name == config]
    if not cfgs:
        console.print(f"[red]Task '{task}' does not declare config '{config}'[/red]")
        raise typer.Exit(1)
    cell = run_one_cell(t, cfgs[0], batch, iterations, seed, per_cell_budget_s)
    console.print_json(json.dumps(cell.to_dict()))


@app.command()
def inspect(path: Path = typer.Argument(..., exists=True)):
    """Pretty-print a results JSON as a per-task table."""
    data = json.loads(path.read_text())
    by_task: dict[str, list[dict]] = {}
    for cell in data["cells"]:
        by_task.setdefault(cell["task"], []).append(cell)
    for task_name, cells in by_task.items():
        table = Table(title=task_name)
        table.add_column("config")
        table.add_column("batch", justify="right")
        table.add_column("status")
        table.add_column("mean (ms)", justify="right")
        table.add_column("ops/s", justify="right")
        table.add_column("speedup", justify="right")
        for c in sorted(cells, key=lambda c: (c["config"], c["batch_size"])):
            mean_ms = f"{c['mean_time_s'] * 1000:.3f}" if c.get("mean_time_s") else "-"
            ops = f"{c['throughput_ops_per_sec']:.2e}" if c.get("throughput_ops_per_sec") else "-"
            spd = f"{c['speedup_vs_baseline']:.2f}x" if c.get("speedup_vs_baseline") else "-"
            status_str = c["status"] if c["status"] == "ok" else f"skip:{c.get('skip_reason')}"
            table.add_row(c["config"], str(c["batch_size"]), status_str, mean_ms, ops, spd)
        console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
