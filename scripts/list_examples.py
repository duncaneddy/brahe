#!/usr/bin/env python3
"""List all available examples."""

from typing import Optional

import typer
from rich.table import Table

from _build_utils import EXAMPLES_DIR, check_flags, console


def main(
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
            _, reason, _ = check_flags(rust_file)
            row.append(reason.upper() if reason else "")

        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    typer.run(main)
