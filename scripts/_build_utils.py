"""Shared utilities for Brahe build scripts."""

import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

# Repository root (scripts/ is one level down from repo root)
REPO_ROOT = Path(__file__).parent.parent.resolve()
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
//! rayon = "1.10"
//! ```
"""

console = Console()


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

    Matching priority:
    1. Strip "examples/" prefix if present
    2. Exact path match
    3. Suffix path match (e.g., "keplerian_propagation/file" matches "orbit_propagation/keplerian_propagation/file")
    4. Filename-only match

    Args:
        directory: Root directory to search in
        filename: Base filename (without extension or with partial path)
        extension: File extension (e.g., '.py', '.rs')

    Returns:
        Path to the file if found, None otherwise
    """
    # Strip "examples/" prefix if user provided it
    if filename.startswith("examples/"):
        filename = filename[len("examples/") :]

    # Normalize extension handling
    if filename.endswith(extension):
        filename_with_ext = filename
        filename_no_ext = filename[: -len(extension)]
    else:
        filename_with_ext = f"{filename}{extension}"
        filename_no_ext = filename

    # 1. Try exact path match from directory root
    direct_path = directory / filename_with_ext
    if direct_path.exists():
        return direct_path

    # 2. Try suffix path match - find files where input is a path suffix
    all_files = list(directory.glob(f"**/*{extension}"))
    suffix_matches = [
        f
        for f in all_files
        if str(f.relative_to(directory)) == filename_with_ext
        or str(f.relative_to(directory)).endswith(f"/{filename_with_ext}")
    ]

    if len(suffix_matches) == 1:
        return suffix_matches[0]
    elif len(suffix_matches) > 1:
        return None

    # 3. Fallback to filename-only match (backward compatibility)
    base_name = Path(filename_no_ext).stem
    pattern = f"**/{base_name}{extension}"
    matches = list(directory.glob(pattern))

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return None

    return None


def check_flags(
    file_path: Path,
    enable_ci_only: bool = False,
    enable_slow: bool = False,
    enable_ignore: bool = False,
) -> tuple[bool, str, int | None]:
    """Check if example/plot should be skipped based on FLAGS and return custom TIMEOUT if set.

    Returns:
        tuple[bool, str, int | None]: (should_skip, reason, timeout_seconds)
        timeout_seconds is None if not specified in the file.
    """
    timeout_seconds = None

    try:
        content = file_path.read_text()
        first_lines = "\n".join(content.split("\n")[:10])

        # Parse TIMEOUT if present
        if "TIMEOUT = " in first_lines:
            timeout_match = re.search(r"TIMEOUT = (\d+)", first_lines)
            if timeout_match:
                timeout_seconds = int(timeout_match.group(1))

        # Parse FLAGS if present
        if "FLAGS = [" in first_lines:
            match = re.search(r"FLAGS = \[(.*?)\]", first_lines)
            if match:
                flags_str = match.group(1)
                flags = [f.strip().strip('"').strip("'") for f in flags_str.split(",")]

                if "IGNORE" in flags and not enable_ignore:
                    return True, "ignored", timeout_seconds
                if "CI-ONLY" in flags and not enable_ci_only:
                    return True, "ci-only", timeout_seconds
                if "SLOW" in flags and not enable_slow:
                    return True, "slow", timeout_seconds
    except Exception:
        pass

    return False, "", timeout_seconds


def test_rust_example(
    file_path: Path, verbose: bool = False, timeout: int = 300
) -> tuple[bool, str, str]:
    """Test a single Rust example. Returns (success, stdout, stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rs", delete=False) as tmp:
        deps = RUST_DEPS.replace("%REPO_ROOT%", str(REPO_ROOT))
        tmp.write(deps)
        tmp.write("\n")
        tmp.write(file_path.read_text())
        tmp_path = tmp.name

    try:
        env = os.environ.copy()
        env["RUSTFLAGS"] = "-D warnings"

        result = subprocess.run(
            ["rust-script", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
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
        error_msg = f"Test timed out after {timeout} seconds"
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
    file_path: Path, verbose: bool = False, timeout: int = 180
) -> tuple[bool, str, str]:
    """Test a single Python example. Returns (success, stdout, stderr)."""
    python_exe = REPO_ROOT / ".venv" / "bin" / "python"
    if not python_exe.exists():
        return False, "", f"Virtual environment not found at {python_exe}"

    try:
        result = subprocess.run(
            [str(python_exe), str(file_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
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
        error_msg = f"Test timed out after {timeout} seconds"
        if verbose:
            console.print(f"[red]{error_msg}[/red]")
        return False, "", error_msg
    except Exception as e:
        error_msg = str(e)
        if verbose:
            console.print(f"[red]Error: {error_msg}[/red]")
        return False, "", error_msg


def run_files_parallel(
    files: List[Path],
    test_fn: Callable[[Path, bool, int], Tuple[bool, str, str]],
    check_flags_fn: Callable[[Path, bool, bool, bool], Tuple[bool, str, Optional[int]]],
    verbose: bool,
    default_timeout: int,
    cli_timeout: Optional[int],
    ci_only: bool,
    slow: bool,
    ignore: bool,
    num_workers: int,
    progress: Progress,
    task_id,
    results: TestResults,
) -> None:
    """Run test files in parallel using ProcessPoolExecutor."""
    # Filter files and prepare tasks
    tasks = []
    for file_path in files:
        should_skip, reason, file_timeout = check_flags_fn(
            file_path, ci_only, slow, ignore
        )
        results.total += 1

        if should_skip:
            results.skipped += 1
            if verbose:
                rel_path = file_path.relative_to(REPO_ROOT)
                console.print(f"  {rel_path}...[yellow]SKIP ({reason})[/yellow]")
            progress.update(task_id, advance=1)
        else:
            effective_timeout = (
                cli_timeout
                if cli_timeout is not None
                else (file_timeout or default_timeout)
            )
            tasks.append((file_path, effective_timeout))

    # Run tasks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(test_fn, file_path, False, timeout): (file_path, timeout)
            for file_path, timeout in tasks
        }

        for future in as_completed(future_to_file):
            file_path, timeout = future_to_file[future]
            rel_path = file_path.relative_to(REPO_ROOT)

            try:
                passed, stdout, stderr = future.result()

                if passed:
                    results.passed += 1
                    if verbose:
                        console.print(f"  {rel_path}...[green]PASS[/green]")
                        if stdout:
                            console.print(stdout, style="dim")
                else:
                    results.failed += 1
                    results.failures.append(str(rel_path))
                    results.error_details.append((str(rel_path), stdout, stderr))
                    console.print(f"  {rel_path}...[red]FAIL[/red]")
                    if verbose:
                        if stdout:
                            console.print("[red]STDOUT:[/red]")
                            console.print(stdout)
                        if stderr:
                            console.print("[red]STDERR:[/red]")
                            console.print(stderr)
            except Exception as e:
                results.failed += 1
                results.failures.append(str(rel_path))
                error_msg = f"Worker exception: {str(e)}"
                results.error_details.append((str(rel_path), "", error_msg))
                console.print(f"  {rel_path}...[red]FAIL[/red]")
                if verbose:
                    console.print(f"[red]Error: {error_msg}[/red]")

            progress.update(task_id, advance=1)


def run_plot_file(plot_file: Path, timeout: int) -> Tuple[str, str, str, int]:
    """Run a single plot file and return results."""
    python_exe = REPO_ROOT / ".venv" / "bin" / "python"

    try:
        result = subprocess.run(
            [str(python_exe), str(plot_file)],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env={
                **subprocess.os.environ,
                "BRAHE_FIGURE_OUTPUT_DIR": str(FIGURE_OUTPUT_DIR),
            },
        )
        return (plot_file.name, result.stdout, result.stderr, result.returncode)
    except subprocess.TimeoutExpired:
        error_msg = f"Timed out after {timeout} seconds"
        return (plot_file.name, "", error_msg, 1)
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        return (plot_file.name, "", error_msg, 1)


def make_progress() -> Progress:
    """Create a standard Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
