#!/usr/bin/env python3
"""Show statistics about examples and plots."""

import typer
from rich.table import Table

from _build_utils import EXAMPLES_DIR, PLOTS_DIR, check_flags, console


def main():
    """Show statistics about examples and plots."""
    # Example stats
    rust_files = list(EXAMPLES_DIR.glob("**/*.rs"))
    python_files = list(EXAMPLES_DIR.glob("**/*.py"))
    pairs = sum(1 for rs in rust_files if rs.with_suffix(".py").exists())

    ignored = sum(1 for rs in rust_files if check_flags(rs)[1] == "ignored")
    ci_only = sum(1 for rs in rust_files if check_flags(rs)[1] == "ci-only")
    slow = sum(1 for rs in rust_files if check_flags(rs)[1] == "slow")

    # Plot stats
    plot_files = [f for f in PLOTS_DIR.glob("**/*.py") if "TEMPLATE" not in f.name]

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
    typer.run(main)
