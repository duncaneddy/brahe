#!/usr/bin/env python3
"""List all available plot scripts."""

import typer
from rich.table import Table

from _build_utils import PLOTS_DIR, check_flags, console


def main(
    show_flags: bool = typer.Option(False, "--flags", help="Show FLAGS"),
):
    """List all available plot scripts."""
    plot_files = sorted(PLOTS_DIR.glob("**/*.py"))
    plot_files = [f for f in plot_files if "TEMPLATE" not in f.name]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Plot Script", style="cyan")
    if show_flags:
        table.add_column("Flags", style="yellow")

    for plot_file in plot_files:
        rel_path = plot_file.relative_to(PLOTS_DIR)
        row = [str(rel_path)]
        if show_flags:
            _, reason, _ = check_flags(plot_file)
            row.append(reason.upper() if reason else "")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    typer.run(main)
