#!/usr/bin/env python3
"""Generate all documentation plots and figures."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import typer

from _build_utils import (
    FIGURE_OUTPUT_DIR,
    PLOTS_DIR,
    REPO_ROOT,
    check_flags,
    console,
    make_progress,
    run_plot_file,
)


def main(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    timeout: Optional[int] = typer.Option(
        None, "--timeout", "-t", help="Override timeout in seconds (default: 180s)"
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers (default: min(cpu_count, 8))",
    ),
    serial: bool = typer.Option(
        False, "--serial", help="Run sequentially instead of in parallel"
    ),
    ci_only: bool = typer.Option(False, "--ci-only", help="Include CI-ONLY plots"),
    slow: bool = typer.Option(False, "--slow", help="Include SLOW plots"),
    ignore: bool = typer.Option(False, "--ignore", help="Include IGNORE plots"),
):
    """Generate all documentation plots and figures."""
    console.print("\n[bold blue]Generating Documentation Figures[/bold blue]\n")

    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_files = sorted(PLOTS_DIR.glob("**/*.py"))
    plot_files = [f for f in plot_files if "TEMPLATE" not in f.name]

    if not plot_files:
        console.print("[yellow]No plot files found in plots/[/yellow]\n")
        return

    # Determine number of workers
    if serial:
        num_workers = 1
    elif workers is not None:
        num_workers = max(1, workers)
    else:
        num_workers = min(os.cpu_count() or 4, 8)

    if num_workers > 1:
        console.print(f"[dim]Running with {num_workers} parallel workers[/dim]\n")

    failed_plots = []
    all_outputs = []

    # Prepare tasks with timeouts, respecting FLAGS
    tasks = []
    skipped_count = 0
    for plot_file in plot_files:
        should_skip, reason, file_timeout = check_flags(
            plot_file, ci_only, slow, ignore
        )

        if should_skip:
            skipped_count += 1
            if verbose:
                console.print(f"  {plot_file.name}...[yellow]SKIP ({reason})[/yellow]")
            continue

        effective_timeout = timeout if timeout is not None else (file_timeout or 180)
        tasks.append((plot_file, effective_timeout))

    with make_progress() as progress:
        task_id = progress.add_task("Generating figures...", total=len(tasks))

        if num_workers == 1:
            for plot_file, effective_timeout in tasks:
                plot_name, stdout, stderr, returncode = run_plot_file(
                    plot_file, effective_timeout
                )
                all_outputs.append((plot_name, stdout, stderr, returncode))

                if returncode != 0:
                    failed_plots.append((plot_name, stdout, stderr))

                progress.update(task_id, advance=1)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_plot = {
                    executor.submit(run_plot_file, plot_file, eff_timeout): (
                        plot_file,
                        eff_timeout,
                    )
                    for plot_file, eff_timeout in tasks
                }

                for future in as_completed(future_to_plot):
                    try:
                        plot_name, stdout, stderr, returncode = future.result()
                        all_outputs.append((plot_name, stdout, stderr, returncode))

                        if returncode != 0:
                            failed_plots.append((plot_name, stdout, stderr))
                    except Exception as e:
                        plot_file, _ = future_to_plot[future]
                        error_msg = f"Worker exception: {str(e)}"
                        all_outputs.append((plot_file.name, "", error_msg, 1))
                        failed_plots.append((plot_file.name, "", error_msg))

                    progress.update(task_id, advance=1)

    # Print all captured outputs
    if verbose:
        console.print("\n[bold]Output from all plots:[/bold]")
        for plot_name, stdout, stderr, returncode in all_outputs:
            console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
            console.print(f"[bold cyan]{plot_name}[/bold cyan]")
            console.print(f"[bold cyan]{'=' * 80}[/bold cyan]")

            if stdout.strip():
                console.print("\n[bold]STDOUT:[/bold]")
                console.print(stdout)

            if stderr.strip():
                console.print("\n[bold]STDERR:[/bold]")
                console.print(stderr)

    if failed_plots:
        console.print("\n[bold red]Failed Plot Details:[/bold red]")
        for plot_name, stdout, stderr in failed_plots:
            console.print(f"\n[bold yellow]{'=' * 80}[/bold yellow]")
            console.print(f"[bold cyan]{plot_name}[/bold cyan]")
            console.print(f"[bold yellow]{'=' * 80}[/bold yellow]")

            if stdout.strip():
                console.print("\n[bold]STDOUT:[/bold]")
                console.print(stdout)

            if stderr.strip():
                console.print("\n[bold]STDERR:[/bold]")
                console.print(stderr)

        console.print()
        raise typer.Exit(1)
    else:
        skip_msg = f" ({skipped_count} skipped)" if skipped_count > 0 else ""
        console.print(
            f"\n[green]✓ All figures generated in {FIGURE_OUTPUT_DIR.relative_to(REPO_ROOT)}{skip_msg}[/green]\n"
        )


if __name__ == "__main__":
    typer.run(main)
