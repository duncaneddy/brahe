#!/usr/bin/env python3
"""Generate a specific plot."""

import subprocess
from pathlib import Path
from typing import Optional

import typer

from _build_utils import (
    FIGURE_OUTPUT_DIR,
    PLOTS_DIR,
    REPO_ROOT,
    check_flags,
    console,
    find_file_by_name,
)


def main(
    plot_name: str = typer.Argument(
        ...,
        help="Plot name (e.g., 'attitude_representations', 'subdir/plot_name', or 'attitude_representations.py')",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    timeout: Optional[int] = typer.Option(
        None, "--timeout", "-t", help="Override timeout in seconds (default: 180s)"
    ),
):
    """Generate a specific plot."""
    # Handle different input formats
    if plot_name.startswith("plots/"):
        plot_name = plot_name[6:]
    if plot_name.endswith(".py"):
        plot_name = plot_name[:-3]

    plot_file = find_file_by_name(PLOTS_DIR, plot_name, ".py")

    if plot_file is None:
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

    _, _, file_timeout = check_flags(plot_file)
    effective_timeout = timeout if timeout is not None else (file_timeout or 180)

    python_exe = REPO_ROOT / ".venv" / "bin" / "python"
    try:
        result = subprocess.run(
            [str(python_exe), str(plot_file)],
            cwd=REPO_ROOT,
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            timeout=effective_timeout,
            env={
                **subprocess.os.environ,
                "BRAHE_FIGURE_OUTPUT_DIR": str(FIGURE_OUTPUT_DIR),
            },
        )
    except subprocess.TimeoutExpired:
        console.print(
            f"[red]✗ Plot generation timed out after {effective_timeout} seconds[/red]\n"
        )
        raise typer.Exit(1)

    if result.returncode != 0:
        console.print(f"[red]✗ Failed to generate {plot_file.name}[/red]\n")

        console.print("[bold yellow]" + "=" * 80 + "[/bold yellow]")
        console.print("[bold cyan]Error Details[/bold cyan]")
        console.print("[bold yellow]" + "=" * 80 + "[/bold yellow]")

        if result.stderr.strip():
            console.print("\n[bold]STDERR:[/bold]")
            console.print(result.stderr)

        console.print()
        raise typer.Exit(1)

    console.print(
        f"[green]✓ Figure generated in {FIGURE_OUTPUT_DIR.relative_to(REPO_ROOT)}[/green]\n"
    )


if __name__ == "__main__":
    typer.run(main)
