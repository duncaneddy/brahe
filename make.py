#!/usr/bin/env python3
# /// script
# dependencies = ["typer", "rich"]
# ///
"""
Brahe Build Tool - Python-based alternative to Makefile.

Provides all Makefile functionality with rich output and better UX.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

app = typer.Typer(
    name="make",
    help="Brahe Build Tool - Test, format, lint, and build",
    add_completion=False,
)
console = Console()

# Repository root
REPO_ROOT = Path(__file__).parent.resolve()
EXAMPLES_DIR = REPO_ROOT / "examples"
PLOTS_DIR = REPO_ROOT / "plots"
FIGURE_OUTPUT_DIR = REPO_ROOT / "docs" / "figures"
TESTS_DIR = REPO_ROOT / "tests"

# Rust dependencies to inject into examples
RUST_DEPS = """//! ```cargo
//! [dependencies]
//! brahe = {path = "%REPO_ROOT%"}
//! nalgebra = { version = "0.34.0", features = ["serde-serialize"] }
//! approx = "^0.5.0"
//! serde_json = "1"
//! ```
"""


class TestResults:
    """Track test results."""

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: List[str] = []
        self.error_details: List[tuple[str, str, str]] = []  # (file, stdout, stderr)


def run_command(
    cmd: List[str], cwd: Optional[Path] = None, verbose: bool = False
) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or REPO_ROOT,
            capture_output=not verbose,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        if verbose:
            console.print(f"[red]Error: {e}[/red]")
        return False


def find_file_by_name(directory: Path, filename: str, extension: str) -> Optional[Path]:
    """
    Find a file by name in directory (including subdirectories).

    Args:
        directory: Root directory to search in
        filename: Base filename (without extension or with partial path)
        extension: File extension (e.g., '.py', '.rs')

    Returns:
        Path to the file if found, None otherwise
    """
    # First try direct path (with or without extension)
    if filename.endswith(extension):
        direct_path = directory / filename
    else:
        direct_path = directory / f"{filename}{extension}"

    if direct_path.exists():
        return direct_path

    # Search recursively by filename only
    base_name = Path(filename).stem  # Get filename without extension
    pattern = f"**/{base_name}{extension}"
    matches = list(directory.glob(pattern))

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Multiple matches found - return None and let caller handle it
        return None

    return None


def check_flags(
    file_path: Path,
    enable_ci_only: bool = False,
    enable_slow: bool = False,
    enable_ignore: bool = False,
) -> tuple[bool, str]:
    """Check if example/plot should be skipped based on FLAGS."""
    try:
        content = file_path.read_text()
        first_lines = "\n".join(content.split("\n")[:10])

        if "FLAGS = [" in first_lines:
            import re

            match = re.search(r"FLAGS = \[(.*?)\]", first_lines)
            if match:
                flags_str = match.group(1)
                # Strip whitespace and quotes from each flag
                flags = [f.strip().strip('"').strip("'") for f in flags_str.split(",")]

                if "IGNORE" in flags and not enable_ignore:
                    return True, "ignored"
                if "CI-ONLY" in flags and not enable_ci_only:
                    return True, "ci-only"
                if "SLOW" in flags and not enable_slow:
                    return True, "slow"
    except Exception:
        pass

    return False, ""


def test_rust_example(file_path: Path, verbose: bool = False) -> tuple[bool, str, str]:
    """Test a single Rust example. Returns (success, stdout, stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rs", delete=False) as tmp:
        deps = RUST_DEPS.replace("%REPO_ROOT%", str(REPO_ROOT))
        tmp.write(deps)
        tmp.write("\n")
        tmp.write(file_path.read_text())
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["rust-script", "--toolchain", "nightly", tmp_path],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            if verbose:
                console.print(result.stdout, style="dim")
            return True, result.stdout, result.stderr
        else:
            if verbose:
                console.print("[red]STDOUT:[/red]")
                console.print(result.stdout)
                console.print("[red]STDERR:[/red]")
                console.print(result.stderr)
            return False, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        error_msg = "Test timed out after 60 seconds"
        if verbose:
            console.print(f"[red]{error_msg}[/red]")
        return False, "", error_msg
    except Exception as e:
        error_msg = str(e)
        if verbose:
            console.print(f"[red]Error: {error_msg}[/red]")
        return False, "", error_msg
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_python_example(
    file_path: Path, verbose: bool = False
) -> tuple[bool, str, str]:
    """Test a single Python example. Returns (success, stdout, stderr)."""
    # Use .venv/bin/python directly to avoid uv environment warnings
    python_exe = REPO_ROOT / ".venv" / "bin" / "python"
    if not python_exe.exists():
        return False, "", f"Virtual environment not found at {python_exe}"

    try:
        result = subprocess.run(
            [str(python_exe), str(file_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=REPO_ROOT,
        )

        if result.returncode == 0:
            if verbose:
                console.print(result.stdout, style="dim")
            return True, result.stdout, result.stderr
        else:
            if verbose:
                console.print("[red]STDOUT:[/red]")
                console.print(result.stdout)
                console.print("[red]STDERR:[/red]")
                console.print(result.stderr)
            return False, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        error_msg = "Test timed out after 60 seconds"
        if verbose:
            console.print(f"[red]{error_msg}[/red]")
        return False, "", error_msg
    except Exception as e:
        error_msg = str(e)
        if verbose:
            console.print(f"[red]Error: {error_msg}[/red]")
        return False, "", error_msg


# ===== Testing Commands =====


@app.command()
def test(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """
    Run all tests (Rust + Python).

    Equivalent to: make test
    """
    console.print("\n[bold blue]Running All Tests[/bold blue]\n")

    success = True

    # Run Rust tests
    console.print("[bold]Running Rust Tests[/bold]")
    if not run_command(["cargo", "test"], verbose=verbose):
        success = False
        console.print("[red]✗ Rust tests failed[/red]\n")
    else:
        console.print("[green]✓ Rust tests passed[/green]\n")

    # Run Python tests
    console.print("[bold]Running Python Tests[/bold]")
    run_command(["uv", "pip", "install", "-e", "."], verbose=False)
    if not run_command(["uv", "run", "pytest", "tests/", "-v"], verbose=verbose):
        success = False
        console.print("[red]✗ Python tests failed[/red]\n")
    else:
        console.print("[green]✓ Python tests passed[/green]\n")

    if success:
        console.print("[bold green]✓ All tests passed![/bold green]\n")
    else:
        raise typer.Exit(1)


@app.command()
def test_rust(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """
    Run Rust tests.

    Equivalent to: make test-rust
    """
    console.print("\n[bold blue]Running Rust Tests[/bold blue]\n")
    if not run_command(["cargo", "test"], verbose=verbose):
        console.print("[red]✗ Rust tests failed[/red]\n")
        raise typer.Exit(1)
    console.print("[green]✓ Rust tests passed[/green]\n")


@app.command()
def test_python(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """
    Run Python tests.

    Equivalent to: make test-python
    """
    console.print("\n[bold blue]Running Python Tests[/bold blue]\n")
    run_command(["uv", "pip", "install", "-e", "."], verbose=False)
    if not run_command(["uv", "run", "pytest", "tests/", "-v"], verbose=verbose):
        console.print("[red]✗ Python tests failed[/red]\n")
        raise typer.Exit(1)
    console.print("[green]✓ Python tests passed[/green]\n")


# ===== Example Commands =====


@app.command()
def list_examples(
    filter_str: Optional[str] = typer.Option(
        None, "--filter", "-f", help="Filter by path"
    ),
    show_flags: bool = typer.Option(False, "--flags", help="Show FLAGS"),
):
    """List all available examples."""
    rust_files = sorted(EXAMPLES_DIR.glob("**/*.rs"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Example", style="cyan")
    table.add_column("Python", justify="center")
    table.add_column("Rust", justify="center")
    if show_flags:
        table.add_column("Flags", style="yellow")

    for rust_file in rust_files:
        rel_path = rust_file.relative_to(EXAMPLES_DIR)
        example_name = str(rel_path.with_suffix(""))

        if filter_str and filter_str not in example_name:
            continue

        py_file = rust_file.with_suffix(".py")
        has_python = "✓" if py_file.exists() else "✗"
        has_rust = "✓"

        row = [example_name, has_python, has_rust]

        if show_flags:
            _, reason = check_flags(rust_file)
            row.append(reason.upper() if reason else "")

        table.add_row(*row)

    console.print(table)


@app.command()
def test_examples(
    strict: bool = typer.Option(False, "--strict", help="Fail on parity issues"),
    ci_only: bool = typer.Option(False, "--ci-only", help="Include CI-ONLY examples"),
    slow: bool = typer.Option(False, "--slow", help="Include SLOW examples"),
    ignore: bool = typer.Option(False, "--ignore", help="Include IGNORE examples"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    lang: Optional[str] = typer.Option(
        None, "--lang", help="Filter by language: python/py or rust/rs"
    ),
):
    """
    Test all documentation examples.

    Equivalent to: make test-examples
    With --strict: make test-examples-strict
    With --ci-only: make test-examples-all
    With --ignore: Include tests marked with IGNORE flag
    With --lang: Test only specified language (python/py or rust/rs)
    """
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

    # Test Rust examples
    rust_results = TestResults()
    if test_rust:
        console.print("[bold]Testing Rust Examples[/bold]")
        rust_files = sorted(EXAMPLES_DIR.glob("**/*.rs"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Testing Rust examples...", total=len(rust_files))

            for rust_file in rust_files:
                rel_path = rust_file.relative_to(REPO_ROOT)
                should_skip, reason = check_flags(rust_file, ci_only, slow, ignore)

                rust_results.total += 1

                if should_skip:
                    rust_results.skipped += 1
                    if verbose:
                        console.print(
                            f"  {rel_path}...[yellow]SKIP ({reason})[/yellow]"
                        )
                else:
                    passed, stdout, stderr = test_rust_example(rust_file, verbose)

                    if passed:
                        rust_results.passed += 1
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

        console.print(
            f"[blue]Rust: {rust_results.total} total, {rust_results.passed} passed, "
            f"{rust_results.failed} failed, {rust_results.skipped} skipped[/blue]\n"
        )

    # Test Python examples
    python_results = TestResults()
    if test_python:
        console.print("[bold]Testing Python Examples[/bold]")
        python_files = sorted(EXAMPLES_DIR.glob("**/*.py"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Testing Python examples...", total=len(python_files)
            )

            for py_file in python_files:
                rel_path = py_file.relative_to(REPO_ROOT)
                should_skip, reason = check_flags(py_file, ci_only, slow, ignore)

                python_results.total += 1

                if should_skip:
                    python_results.skipped += 1
                    if verbose:
                        console.print(
                            f"  {rel_path}...[yellow]SKIP ({reason})[/yellow]"
                        )
                else:
                    passed, stdout, stderr = test_python_example(py_file, verbose)

                    if passed:
                        python_results.passed += 1
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

        # Print detailed error information
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


@app.command()
def test_example(
    example_name: str = typer.Argument(
        ...,
        help="Example name (e.g., 'orbital_period', 'access/basic_workflow', 'orbital_period.py', or 'orbital_period.rs')",
    ),
    lang: Optional[str] = typer.Option(
        None, "--lang", "-l", help="Language: rust, python, or both (default)"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q"),
):
    """
    Test a specific example.

    Equivalent to: make test-example EXAMPLE_NAME=... LANG=...

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
        base_name = example_name[:-3]  # Remove .py extension
        test_rust_lang = False
        test_python_lang = True
    elif example_name.endswith(".rs"):
        base_name = example_name[:-3]  # Remove .rs extension
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
            # Check if multiple matches
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
            # Check if multiple matches
            matches = list(EXAMPLES_DIR.glob(f"**/{Path(base_name).stem}.py"))
            if len(matches) > 1:
                console.print(
                    f"[red]Error: Multiple matches found for '{base_name}.py':[/red]"
                )
                for match in matches:
                    console.print(f"  {match.relative_to(REPO_ROOT)}")
                console.print("[yellow]Please specify the full path[/yellow]")
                raise typer.Exit(1)

    # If neither file found (considering what was requested), handle appropriately
    if test_rust_lang and not rust_file and test_python_lang and not py_file:
        # Looking for both, found neither
        console.print(f"[red]Error: No example files found for '{base_name}'[/red]")
        raise typer.Exit(1)
    elif test_rust_lang and not rust_file and not test_python_lang:
        # Only looking for Rust, didn't find it - but maybe Python exists
        if find_file_by_name(EXAMPLES_DIR, base_name, ".py"):
            console.print(
                f"[red]Error: {base_name}.rs not found (but {base_name}.py exists)[/red]"
            )
        else:
            console.print(f"[red]Error: {base_name}.rs not found[/red]")
        raise typer.Exit(1)
    elif test_python_lang and not py_file and not test_rust_lang:
        # Only looking for Python, didn't find it - but maybe Rust exists
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
        passed, stdout, stderr = test_rust_example(rust_file, verbose)
        if passed:
            console.print("[green]✓ PASS[/green]")
        else:
            console.print("[red]✗ FAIL[/red]")
            all_passed = False
            error_details.append(
                (str(rust_file.relative_to(REPO_ROOT)), stdout, stderr)
            )
    elif test_rust_lang and not rust_file:
        # Warn about missing Rust file
        expected_path = expected_dir / f"{Path(base_name).stem}.rs"
        console.print(
            f"[yellow]⚠ Warning: {expected_path.relative_to(REPO_ROOT)} not found[/yellow]"
        )

    # Test Python if requested and found
    if test_python_lang and py_file:
        console.print(f"[blue]Testing Python: {py_file.relative_to(REPO_ROOT)}[/blue]")
        passed, stdout, stderr = test_python_example(py_file, verbose)
        if passed:
            console.print("[green]✓ PASS[/green]")
        else:
            console.print("[red]✗ FAIL[/red]")
            all_passed = False
            error_details.append((str(py_file.relative_to(REPO_ROOT)), stdout, stderr))
    elif test_python_lang and not py_file:
        # Warn about missing Python file
        expected_path = expected_dir / f"{Path(base_name).stem}.py"
        console.print(
            f"[yellow]⚠ Warning: {expected_path.relative_to(REPO_ROOT)} not found[/yellow]"
        )

    if not all_passed:
        # Print detailed error information
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


# ===== Plot/Figure Commands =====


@app.command()
def make_plots(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """
    Generate all documentation plots and figures.
    """
    console.print("\n[bold blue]Generating Documentation Figures[/bold blue]\n")

    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_files = sorted(PLOTS_DIR.glob("*.py"))
    if not plot_files:
        console.print("[yellow]No plot files found in plots/[/yellow]\n")
        return

    python_exe = REPO_ROOT / ".venv" / "bin" / "python"
    failed_plots = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating figures...", total=len(plot_files))

        for plot_file in plot_files:
            if verbose:
                console.print(f"Generating {plot_file.name}...")

            result = subprocess.run(
                [str(python_exe), str(plot_file)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                env={
                    **subprocess.os.environ,
                    "BRAHE_FIGURE_OUTPUT_DIR": str(FIGURE_OUTPUT_DIR),
                },
            )

            if result.returncode != 0:
                failed_plots.append((plot_file.name, result.stdout, result.stderr))
                console.print(f"[red]✗ Failed to generate {plot_file.name}[/red]")

            progress.update(task, advance=1)

    # Print detailed error information for failed plots
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
        console.print(
            f"\n[green]✓ All figures generated in {FIGURE_OUTPUT_DIR.relative_to(REPO_ROOT)}[/green]\n"
        )


@app.command()
def make_plot(
    plot_name: str = typer.Argument(
        ...,
        help="Plot name (e.g., 'attitude_representations', 'subdir/plot_name', or 'attitude_representations.py')",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Generate a specific plot.

    Equivalent to: make plot NAME=...

    The plot name can be:
    - Just the filename (searches all subdirectories): 'attitude_representations'
    - Full path from plots/: 'subdir/plot_name'
    - With .py extension: 'attitude_representations.py'
    """
    # Handle different input formats
    # Remove leading "plots/" if present
    if plot_name.startswith("plots/"):
        plot_name = plot_name[6:]
    # Remove trailing .py if present
    if plot_name.endswith(".py"):
        plot_name = plot_name[:-3]

    # Try to find the plot file
    plot_file = find_file_by_name(PLOTS_DIR, plot_name, ".py")

    if plot_file is None:
        # Check if multiple matches
        matches = list(PLOTS_DIR.glob(f"**/{Path(plot_name).stem}.py"))
        if len(matches) > 1:
            console.print(
                f"[red]Error: Multiple matches found for '{plot_name}.py':[/red]"
            )
            for match in matches:
                console.print(f"  {match.relative_to(REPO_ROOT)}")
            console.print("[yellow]Please specify the full path[/yellow]")
        else:
            console.print(f"[red]Error: {plot_name}.py not found in plots/[/red]")
            console.print("\n[yellow]Available plots:[/yellow]")
            for p in sorted(PLOTS_DIR.glob("**/*.py")):
                console.print(f"  {p.relative_to(PLOTS_DIR).with_suffix('')}")
        console.print()
        raise typer.Exit(1)

    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold blue]Generating {plot_file.name}[/bold blue]\n")

    python_exe = REPO_ROOT / ".venv" / "bin" / "python"
    result = subprocess.run(
        [str(python_exe), str(plot_file)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            **subprocess.os.environ,
            "BRAHE_FIGURE_OUTPUT_DIR": str(FIGURE_OUTPUT_DIR),
        },
    )

    if result.returncode != 0:
        console.print(f"[red]✗ Failed to generate {plot_file.name}[/red]\n")

        console.print("[bold yellow]" + "=" * 80 + "[/bold yellow]")
        console.print("[bold cyan]Error Details[/bold cyan]")
        console.print("[bold yellow]" + "=" * 80 + "[/bold yellow]")

        if result.stdout.strip():
            console.print("\n[bold]STDOUT:[/bold]")
            console.print(result.stdout)

        if result.stderr.strip():
            console.print("\n[bold]STDERR:[/bold]")
            console.print(result.stderr)

        console.print()
        raise typer.Exit(1)

    console.print(
        f"[green]✓ Figure generated in {FIGURE_OUTPUT_DIR.relative_to(REPO_ROOT)}[/green]\n"
    )


@app.command()
def list_plots(show_flags: bool = typer.Option(False, "--flags")):
    """List all available plot scripts."""
    plot_files = sorted(PLOTS_DIR.glob("*.py"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Plot Script", style="cyan")
    if show_flags:
        table.add_column("Flags", style="yellow")

    for plot_file in plot_files:
        row = [plot_file.name]
        if show_flags:
            _, reason = check_flags(plot_file)
            row.append(reason.upper() if reason else "")
        table.add_row(*row)

    console.print(table)


# ===== Code Quality Commands =====


@app.command()
def format(
    check: bool = typer.Option(
        False, "--check", help="Check formatting without changes"
    ),
):
    """
    Format all code (Rust + Python).

    Equivalent to: make format (or make format-check with --check)
    """
    console.print("\n[bold blue]Formatting Code[/bold blue]\n")

    # Rust
    console.print("[bold]Formatting Rust[/bold]")
    cmd = ["cargo", "fmt"]
    if check:
        cmd.extend(["--", "--check"])

    if not run_command(cmd, verbose=True):
        console.print("[red]✗ Rust formatting failed[/red]\n")
        raise typer.Exit(1)
    console.print("[green]✓ Rust formatted[/green]\n")

    # Python
    console.print("[bold]Formatting Python[/bold]")
    cmd = ["uv", "run", "ruff", "format"]
    if check:
        cmd.append("--check")

    if not run_command(cmd, verbose=True):
        console.print("[red]✗ Python formatting failed[/red]\n")
        raise typer.Exit(1)
    console.print("[green]✓ Python formatted[/green]\n")


@app.command()
def lint(
    fix: bool = typer.Option(False, "--fix", help="Auto-fix issues where possible"),
):
    """
    Run linters (clippy + ruff).

    Equivalent to: make lint (or make lint-fix with --fix)
    """
    console.print("\n[bold blue]Running Linters[/bold blue]\n")

    # Rust clippy
    console.print("[bold]Running Clippy[/bold]")
    if not run_command(
        ["cargo", "clippy", "--all-targets", "--all-features", "--", "-D", "warnings"],
        verbose=True,
    ):
        console.print("[red]✗ Clippy found issues[/red]\n")
        raise typer.Exit(1)
    console.print("[green]✓ Clippy passed[/green]\n")

    # Python ruff
    console.print("[bold]Running Ruff[/bold]")
    cmd = ["uv", "run", "ruff", "check"]
    if fix:
        cmd.append("--fix")

    if not run_command(cmd, verbose=True):
        console.print("[red]✗ Ruff found issues[/red]\n")
        raise typer.Exit(1)
    console.print("[green]✓ Ruff passed[/green]\n")


# ===== Documentation Commands =====


@app.command()
def docs(
    serve: bool = typer.Option(False, "--serve", help="Serve docs locally"),
    strict: bool = typer.Option(
        False, "--strict", help="Strict mode (warnings as errors)"
    ),
):
    """
    Build documentation.

    Equivalent to: make build-docs (or make serve-docs with --serve)
    """
    console.print("\n[bold blue]Building Documentation[/bold blue]\n")

    # Generate figures first
    console.print("[bold]Generating figures[/bold]")
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    python_exe = REPO_ROOT / ".venv" / "bin" / "python"
    for plot_file in PLOTS_DIR.glob("*.py"):
        subprocess.run(
            [str(python_exe), str(plot_file)],
            cwd=REPO_ROOT,
            capture_output=True,
            env={
                **subprocess.os.environ,
                "BRAHE_FIGURE_OUTPUT_DIR": str(FIGURE_OUTPUT_DIR),
            },
        )
    console.print("[green]✓ Figures generated[/green]\n")

    # Generate stubs
    console.print("[bold]Generating type stubs[/bold]")
    if not run_command(["./scripts/generate_stubs.sh"]):
        console.print("[red]✗ Stub generation failed[/red]\n")
        raise typer.Exit(1)
    console.print("[green]✓ Stubs generated[/green]\n")

    # Build or serve docs
    if serve:
        console.print("[bold]Serving documentation at http://127.0.0.1:8000[/bold]\n")
        subprocess.run(
            ["uv", "run", "mkdocs", "serve"],
            cwd=REPO_ROOT / "docs",
        )
    else:
        console.print("[bold]Building documentation[/bold]")
        cmd = ["uv", "run", "mkdocs", "build"]
        if strict:
            cmd.append("--strict")

        if not run_command(cmd, cwd=REPO_ROOT / "docs", verbose=True):
            console.print("[red]✗ Documentation build failed[/red]\n")
            raise typer.Exit(1)
        console.print("[green]✓ Documentation built[/green]\n")


# ===== Utility Commands =====


@app.command()
def clean():
    """
    Clean build artifacts.

    Equivalent to: make clean
    """
    console.print("\n[bold blue]Cleaning Build Artifacts[/bold blue]\n")

    run_command(["cargo", "clean"])

    dirs_to_remove = [
        REPO_ROOT / "target",
        REPO_ROOT / "docs" / "site",
    ]

    for dir_path in dirs_to_remove:
        if dir_path.exists():
            import shutil

            shutil.rmtree(dir_path)
            console.print(f"Removed {dir_path.relative_to(REPO_ROOT)}")

    # Remove HTML figures
    for html_file in FIGURE_OUTPUT_DIR.glob("*.html"):
        html_file.unlink()

    console.print("\n[green]✓ Cleaned![/green]\n")


@app.command()
def stats():
    """
    Show statistics about examples and plots.
    """
    # Example stats
    rust_files = list(EXAMPLES_DIR.glob("**/*.rs"))
    python_files = list(EXAMPLES_DIR.glob("**/*.py"))
    pairs = sum(1 for rs in rust_files if rs.with_suffix(".py").exists())

    ignored = sum(1 for rs in rust_files if check_flags(rs)[1] == "ignored")
    ci_only = sum(1 for rs in rust_files if check_flags(rs)[1] == "ci-only")
    slow = sum(1 for rs in rust_files if check_flags(rs)[1] == "slow")

    # Plot stats
    plot_files = list(PLOTS_DIR.glob("*.py"))

    console.print("\n[bold]Statistics[/bold]\n")

    # Examples table
    console.print("[bold cyan]Examples[/bold cyan]")
    table = Table(show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="bold")

    table.add_row("Total Rust examples", str(len(rust_files)))
    table.add_row("Total Python examples", str(len(python_files)))
    table.add_row("Complete pairs", str(pairs))
    table.add_row("Missing Python", str(len(rust_files) - pairs))
    table.add_row("Missing Rust", str(len(python_files) - pairs))
    table.add_row("", "")
    table.add_row("IGNORE flagged", str(ignored))
    table.add_row("CI-ONLY flagged", str(ci_only))
    table.add_row("SLOW flagged", str(slow))

    console.print(table)

    # Plots table
    console.print("\n[bold cyan]Plots[/bold cyan]")
    plot_table = Table(show_header=False)
    plot_table.add_column("Metric", style="cyan")
    plot_table.add_column("Count", justify="right", style="bold")

    plot_table.add_row("Total plot scripts", str(len(plot_files)))

    console.print(plot_table)
    console.print()


if __name__ == "__main__":
    app()
