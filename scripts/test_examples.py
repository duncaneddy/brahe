#!/usr/bin/env python3
"""Test all documentation examples."""

import os
from typing import Optional

import typer
from rich.table import Table

from _build_utils import (
    EXAMPLES_DIR,
    REPO_ROOT,
    TestResults,
    check_flags,
    console,
    make_progress,
    run_files_parallel,
    save_example_output,
    test_python_example,
    test_rust_example,
)


def main(
    strict: bool = typer.Option(False, "--strict", help="Fail on parity issues"),
    ci_only: bool = typer.Option(False, "--ci-only", help="Include CI-ONLY examples"),
    slow: bool = typer.Option(False, "--slow", help="Include SLOW examples"),
    ignore: bool = typer.Option(False, "--ignore", help="Include IGNORE examples"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    lang: Optional[str] = typer.Option(
        None, "--lang", help="Filter by language: python/py or rust/rs"
    ),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        "-t",
        help="Override timeout in seconds (default: Python=180s, Rust=300s)",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers (default: cpu_count - 1)",
    ),
    serial: bool = typer.Option(
        False, "--serial", help="Run sequentially instead of in parallel"
    ),
):
    """Test all documentation examples."""
    # Normalize language input
    test_rust = True
    test_python = True

    if lang:
        lang_lower = lang.lower()
        if lang_lower in ["python", "py"]:
            test_rust = False
            test_python = True
        elif lang_lower in ["rust", "rs"]:
            test_rust = True
            test_python = False
        else:
            console.print(
                f"[red]Error: Invalid language '{lang}'. Use: python, py, rust, or rs[/red]\n"
            )
            raise typer.Exit(1)

    lang_filter_msg = ""
    if not test_rust:
        lang_filter_msg = " (Python only)"
    elif not test_python:
        lang_filter_msg = " (Rust only)"

    console.print(
        f"\n[bold blue]Testing Documentation Examples{lang_filter_msg}[/bold blue]\n"
    )

    # Determine number of workers
    if serial:
        rust_workers = 1
        python_workers = 1
    elif workers is not None:
        rust_workers = max(1, workers)
        python_workers = max(1, workers)
    else:
        cpu_count = os.cpu_count() or 1
        rust_workers = max(cpu_count - 1, 1)
        python_workers = max(cpu_count - 1, 1)

    # Test Rust examples
    rust_results = TestResults()
    if test_rust:
        console.print("[bold]Testing Rust Examples[/bold]")
        if rust_workers > 1:
            console.print(f"[dim]Running with {rust_workers} parallel workers[/dim]")

        rust_files = sorted(EXAMPLES_DIR.glob("**/*.rs"))

        with make_progress() as progress:
            task = progress.add_task("Testing Rust examples...", total=len(rust_files))

            if rust_workers == 1:
                for rust_file in rust_files:
                    rel_path = rust_file.relative_to(REPO_ROOT)
                    should_skip, reason, file_timeout = check_flags(
                        rust_file, ci_only, slow, ignore
                    )

                    rust_results.total += 1

                    if should_skip:
                        rust_results.skipped += 1
                        if verbose:
                            console.print(
                                f"  {rel_path}...[yellow]SKIP ({reason})[/yellow]"
                            )
                    else:
                        effective_timeout = (
                            timeout if timeout is not None else (file_timeout or 300)
                        )
                        passed, stdout, stderr = test_rust_example(
                            rust_file, verbose, effective_timeout
                        )

                        if passed:
                            rust_results.passed += 1
                            save_example_output(rust_file, stdout)
                            if verbose:
                                console.print(f"  {rel_path}...[green]PASS[/green]")
                        else:
                            rust_results.failed += 1
                            rust_results.failures.append(str(rel_path))
                            rust_results.error_details.append(
                                (str(rel_path), stdout, stderr)
                            )
                            console.print(f"  {rel_path}...[red]FAIL[/red]")

                    progress.update(task, advance=1)
            else:
                run_files_parallel(
                    rust_files,
                    test_rust_example,
                    check_flags,
                    verbose,
                    300,
                    timeout,
                    ci_only,
                    slow,
                    ignore,
                    rust_workers,
                    progress,
                    task,
                    rust_results,
                    on_pass=save_example_output,
                )

        console.print(
            f"[blue]Rust: {rust_results.total} total, {rust_results.passed} passed, "
            f"{rust_results.failed} failed, {rust_results.skipped} skipped[/blue]\n"
        )

    # Test Python examples
    python_results = TestResults()
    if test_python:
        console.print("[bold]Testing Python Examples[/bold]")
        if python_workers > 1:
            console.print(f"[dim]Running with {python_workers} parallel workers[/dim]")

        python_files = sorted(EXAMPLES_DIR.glob("**/*.py"))

        with make_progress() as progress:
            task = progress.add_task(
                "Testing Python examples...", total=len(python_files)
            )

            if python_workers == 1:
                for py_file in python_files:
                    rel_path = py_file.relative_to(REPO_ROOT)
                    should_skip, reason, file_timeout = check_flags(
                        py_file, ci_only, slow, ignore
                    )

                    python_results.total += 1

                    if should_skip:
                        python_results.skipped += 1
                        if verbose:
                            console.print(
                                f"  {rel_path}...[yellow]SKIP ({reason})[/yellow]"
                            )
                    else:
                        effective_timeout = (
                            timeout if timeout is not None else (file_timeout or 180)
                        )
                        passed, stdout, stderr = test_python_example(
                            py_file, verbose, effective_timeout
                        )

                        if passed:
                            python_results.passed += 1
                            save_example_output(py_file, stdout)
                            if verbose:
                                console.print(f"  {rel_path}...[green]PASS[/green]")
                        else:
                            python_results.failed += 1
                            python_results.failures.append(str(rel_path))
                            python_results.error_details.append(
                                (str(rel_path), stdout, stderr)
                            )
                            console.print(f"  {rel_path}...[red]FAIL[/red]")

                    progress.update(task, advance=1)
            else:
                run_files_parallel(
                    python_files,
                    test_python_example,
                    check_flags,
                    verbose,
                    180,
                    timeout,
                    ci_only,
                    slow,
                    ignore,
                    python_workers,
                    progress,
                    task,
                    python_results,
                    on_pass=save_example_output,
                )

        console.print(
            f"[blue]Python: {python_results.total} total, {python_results.passed} passed, "
            f"{python_results.failed} failed, {python_results.skipped} skipped[/blue]\n"
        )

    # Check parity (only when testing both languages)
    if test_rust and test_python:
        console.print("[bold]Checking Rust/Python Parity[/bold]")
        missing_py = []
        missing_rs = []

        for rs_file in EXAMPLES_DIR.glob("**/*.rs"):
            py_file = rs_file.with_suffix(".py")
            if not py_file.exists():
                missing_py.append(rs_file.relative_to(REPO_ROOT))

        for py_file in EXAMPLES_DIR.glob("**/*.py"):
            rs_file = py_file.with_suffix(".rs")
            if not rs_file.exists():
                missing_rs.append(py_file.relative_to(REPO_ROOT))

        if missing_py or missing_rs:
            console.print(
                f"[yellow]Parity Issues: {len(missing_py)} missing Python, "
                f"{len(missing_rs)} missing Rust[/yellow]"
            )
            if strict:
                console.print("\n[red]STRICT MODE: Failing due to parity issues[/red]")
                raise typer.Exit(1)
        else:
            console.print("[green]All examples have Rust/Python pairs![/green]")

    # Summary
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Language", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Skipped", justify="right", style="yellow")

    if test_rust:
        table.add_row(
            "Rust",
            str(rust_results.total),
            str(rust_results.passed),
            str(rust_results.failed),
            str(rust_results.skipped),
        )
    if test_python:
        table.add_row(
            "Python",
            str(python_results.total),
            str(python_results.passed),
            str(python_results.failed),
            str(python_results.skipped),
        )

    if rust_results.failures or python_results.failures:
        console.print("\n[bold red]Failed Examples:[/bold red]")
        for failure in rust_results.failures + python_results.failures:
            console.print(f"  [red]✗[/red] {failure}")

        console.print("\n[bold red]Error Details:[/bold red]")
        all_errors = rust_results.error_details + python_results.error_details
        for file_path, stdout, stderr in all_errors:
            console.print(f"\n[bold yellow]{'=' * 80}[/bold yellow]")
            console.print(f"[bold cyan]{file_path}[/bold cyan]")
            console.print(f"[bold yellow]{'=' * 80}[/bold yellow]")

            if stdout.strip():
                console.print("\n[bold]STDOUT:[/bold]")
                console.print(stdout)

            if stderr.strip():
                console.print("\n[bold]STDERR:[/bold]")
                console.print(stderr)

        console.print()

    console.print("\n[bold]Summary[/bold]")
    console.print(table)

    if rust_results.failures or python_results.failures:
        console.print("\n[bold red]Failed Examples:[/bold red]")
        for failure in rust_results.failures + python_results.failures:
            console.print(f"  [red]✗[/red] {failure}")

        raise typer.Exit(1)
    else:
        console.print("\n[bold green]✓ All tests passed![/bold green]\n")


if __name__ == "__main__":
    typer.run(main)
