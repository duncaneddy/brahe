#!/usr/bin/env python3
"""Strip output comments from example files.

Removes blocks starting with markers like "# Expected output:", "# Output:",
"# Outputs:" (Python) or their // equivalents (Rust) and all subsequent
comment lines through the end of the block.

Only strips blocks preceded by a recognized marker. Unmarked inline comments
are left untouched.
"""

import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()

REPO_ROOT = Path(__file__).parent.parent.resolve()
EXAMPLES_DIR = REPO_ROOT / "examples"

# Markers that identify the start of an output block
PYTHON_MARKER = re.compile(r"^# (?:Expected )?[Oo]utput(?:s)?:\s*$")
PYTHON_INLINE_MARKER = re.compile(r"^# (?:Expected )?[Oo]utput(?:s)?:\s+\S")
PYTHON_COMMENT = re.compile(r"^#")

RUST_MARKER = re.compile(r"^\s*// (?:Expected )?[Oo]utput(?:s)?:\s*$")
RUST_INLINE_MARKER = re.compile(r"^\s*// (?:Expected )?[Oo]utput(?:s)?:\s+\S")
RUST_COMMENT = re.compile(r"^\s*//")


def strip_python_output(content: str) -> str:
    """Strip output comment blocks from Python example content."""
    lines = content.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for inline marker (e.g., "# Output: True") - remove just that line
        if PYTHON_INLINE_MARKER.match(line):
            i += 1
            continue

        # Check for block marker (e.g., "# Expected output:")
        if PYTHON_MARKER.match(line):
            # Skip the marker and all subsequent comment/blank lines
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped == "" or PYTHON_COMMENT.match(stripped):
                    i += 1
                else:
                    break
            continue

        result.append(line)
        i += 1

    # Remove trailing blank lines
    while result and result[-1].strip() == "":
        result.pop()

    return "\n".join(result) + "\n"


def strip_rust_output(content: str) -> str:
    """Strip output comment blocks from Rust example content."""
    lines = content.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for inline marker (e.g., "// Output: True") - remove just that line
        if RUST_INLINE_MARKER.match(line):
            i += 1
            continue

        # Check for block marker (e.g., "// Expected output:")
        if RUST_MARKER.match(line):
            # Skip the marker and all subsequent comment/blank lines
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped == "" or RUST_COMMENT.match(stripped):
                    i += 1
                else:
                    break
            continue

        result.append(line)
        i += 1

    # Remove trailing blank lines (but preserve closing brace)
    while len(result) > 1 and result[-2].strip() == "" and result[-1].strip() == "}":
        result.pop(-2)

    return "\n".join(result) + "\n"


def process_file(file_path: Path, dry_run: bool) -> bool:
    """Process a single file. Returns True if changes were made."""
    content = file_path.read_text()

    if file_path.suffix == ".py":
        new_content = strip_python_output(content)
    elif file_path.suffix == ".rs":
        new_content = strip_rust_output(content)
    else:
        return False

    if content == new_content:
        return False

    if dry_run:
        console.print(
            f"[yellow]Would modify:[/yellow] {file_path.relative_to(REPO_ROOT)}"
        )
    else:
        file_path.write_text(new_content)
        console.print(f"[green]Modified:[/green] {file_path.relative_to(REPO_ROOT)}")

    return True


def main(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would change without writing"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show unchanged files too"
    ),
    example: Optional[str] = typer.Option(
        None, "--example", "-e", help="Process a single example by name"
    ),
):
    """Strip output comments from example files."""
    if example:
        # Process a single example
        py_file = EXAMPLES_DIR / f"{example}.py"
        rs_file = EXAMPLES_DIR / f"{example}.rs"
        files = [f for f in [py_file, rs_file] if f.exists()]
        if not files:
            console.print(f"[red]No files found for '{example}'[/red]")
            raise typer.Exit(1)
    else:
        # Process all examples
        files = sorted(
            list(EXAMPLES_DIR.glob("**/*.py")) + list(EXAMPLES_DIR.glob("**/*.rs"))
        )

    modified = 0
    unchanged = 0

    for file_path in files:
        changed = process_file(file_path, dry_run)
        if changed:
            modified += 1
        else:
            unchanged += 1
            if verbose:
                console.print(
                    f"[dim]Unchanged: {file_path.relative_to(REPO_ROOT)}[/dim]"
                )

    action = "Would modify" if dry_run else "Modified"
    console.print(
        f"\n[bold]{action}: {modified} files, Unchanged: {unchanged} files[/bold]"
    )


if __name__ == "__main__":
    typer.run(main)
