#!/usr/bin/env python3
"""Test a specific example."""

from pathlib import Path
from typing import Optional

import typer

from _build_utils import (
    EXAMPLES_DIR,
    REPO_ROOT,
    check_flags,
    console,
    find_file_by_name,
    test_python_example,
    test_rust_example,
)


def main(
    example_name: str = typer.Argument(
        ...,
        help="Example name (e.g., 'orbital_period', 'access/basic_workflow', 'orbital_period.py', or 'orbital_period.rs')",
    ),
    lang: Optional[str] = typer.Option(
        None, "--lang", "-l", help="Language: rust, python, or both (default)"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q"),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        "-t",
        help="Override timeout in seconds (default: Python=180s, Rust=300s)",
    ),
):
    """
    Test a specific example.

    The example name can be:
    - Just the filename (searches all subdirectories): 'orbital_period'
    - Full path from examples/: 'common/orbital_period'
    - With extension for single language: 'orbital_period.py' or 'orbital_period.rs'

    If example_name ends with .py, only Python test runs.
    If example_name ends with .rs, only Rust test runs.
    If example_name has no extension, both tests run (unless --lang is specified).
    """
    # Detect file extension and adjust behavior
    if example_name.endswith(".py"):
        base_name = example_name[:-3]
        test_rust_lang = False
        test_python_lang = True
    elif example_name.endswith(".rs"):
        base_name = example_name[:-3]
        test_rust_lang = True
        test_python_lang = False
    else:
        base_name = example_name
        test_rust_lang = lang in (None, "both", "rust")
        test_python_lang = lang in (None, "both", "python")

    all_passed = True
    error_details = []
    rust_file = None
    py_file = None

    # Try to find both files first
    if test_rust_lang:
        rust_file = find_file_by_name(EXAMPLES_DIR, base_name, ".rs")
        if rust_file is None:
            matches = list(EXAMPLES_DIR.glob(f"**/{Path(base_name).stem}.rs"))
            if len(matches) > 1:
                console.print(
                    f"[red]Error: Multiple matches found for '{base_name}.rs':[/red]"
                )
                for match in matches:
                    console.print(f"  {match.relative_to(REPO_ROOT)}")
                console.print("[yellow]Please specify the full path[/yellow]")
                raise typer.Exit(1)

    if test_python_lang:
        py_file = find_file_by_name(EXAMPLES_DIR, base_name, ".py")
        if py_file is None:
            matches = list(EXAMPLES_DIR.glob(f"**/{Path(base_name).stem}.py"))
            if len(matches) > 1:
                console.print(
                    f"[red]Error: Multiple matches found for '{base_name}.py':[/red]"
                )
                for match in matches:
                    console.print(f"  {match.relative_to(REPO_ROOT)}")
                console.print("[yellow]Please specify the full path[/yellow]")
                raise typer.Exit(1)

    # If neither file found, handle appropriately
    if test_rust_lang and not rust_file and test_python_lang and not py_file:
        console.print(f"[red]Error: No example files found for '{base_name}'[/red]")
        raise typer.Exit(1)
    elif test_rust_lang and not rust_file and not test_python_lang:
        if find_file_by_name(EXAMPLES_DIR, base_name, ".py"):
            console.print(
                f"[red]Error: {base_name}.rs not found (but {base_name}.py exists)[/red]"
            )
        else:
            console.print(f"[red]Error: {base_name}.rs not found[/red]")
        raise typer.Exit(1)
    elif test_python_lang and not py_file and not test_rust_lang:
        if find_file_by_name(EXAMPLES_DIR, base_name, ".rs"):
            console.print(
                f"[red]Error: {base_name}.py not found (but {base_name}.rs exists)[/red]"
            )
        else:
            console.print(f"[red]Error: {base_name}.py not found[/red]")
        raise typer.Exit(1)

    # Determine expected path for missing file warnings
    if rust_file:
        expected_dir = rust_file.parent
    elif py_file:
        expected_dir = py_file.parent
    else:
        expected_dir = EXAMPLES_DIR

    # Test Rust if requested and found
    if test_rust_lang and rust_file:
        console.print(f"[blue]Testing Rust: {rust_file.relative_to(REPO_ROOT)}[/blue]")
        should_skip, reason, rust_timeout = check_flags(rust_file)
        if should_skip:
            console.print(f"[yellow]SKIP ({reason})[/yellow]")
        else:
            effective_timeout = (
                timeout if timeout is not None else (rust_timeout or 300)
            )
            passed, stdout, stderr = test_rust_example(
                rust_file, verbose, effective_timeout
            )
            if passed:
                console.print("[green]✓ PASS[/green]")
            else:
                console.print("[red]✗ FAIL[/red]")
                all_passed = False
                error_details.append(
                    (str(rust_file.relative_to(REPO_ROOT)), stdout, stderr)
                )
    elif test_rust_lang and not rust_file:
        expected_path = expected_dir / f"{Path(base_name).stem}.rs"
        console.print(
            f"[yellow]⚠ Warning: {expected_path.relative_to(REPO_ROOT)} not found[/yellow]"
        )

    # Test Python if requested and found
    if test_python_lang and py_file:
        console.print(f"[blue]Testing Python: {py_file.relative_to(REPO_ROOT)}[/blue]")
        should_skip, reason, py_timeout = check_flags(py_file)
        if should_skip:
            console.print(f"[yellow]SKIP ({reason})[/yellow]")
        else:
            effective_timeout = timeout if timeout is not None else (py_timeout or 180)
            passed, stdout, stderr = test_python_example(
                py_file, verbose, effective_timeout
            )
            if passed:
                console.print("[green]✓ PASS[/green]")
            else:
                console.print("[red]✗ FAIL[/red]")
                all_passed = False
                error_details.append(
                    (str(py_file.relative_to(REPO_ROOT)), stdout, stderr)
                )
    elif test_python_lang and not py_file:
        expected_path = expected_dir / f"{Path(base_name).stem}.py"
        console.print(
            f"[yellow]⚠ Warning: {expected_path.relative_to(REPO_ROOT)} not found[/yellow]"
        )

    if not all_passed:
        console.print("\n[bold red]Error Details:[/bold red]")
        for file_path, stdout, stderr in error_details:
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
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
