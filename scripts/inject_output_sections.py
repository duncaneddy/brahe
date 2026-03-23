#!/usr/bin/env python3
"""Inject collapsible output sections into documentation markdown files.

Scans docs/ for tabbed Python/Rust example pairs using --8<-- includes and
injects a ??? example "Output" collapsible section after each pair. The output
section contains synced language tabs that link to captured output text files
in docs/outputs/.

Only processes includes with line number offsets (e.g., :8, :4). Named
segment includes (e.g., :preamble, :all) are skipped since they represent
parts of a larger file whose output belongs at a different location.
"""

import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()

REPO_ROOT = Path(__file__).parent.parent.resolve()
DOCS_DIR = REPO_ROOT / "docs"
OUTPUTS_DIR = REPO_ROOT / "docs" / "outputs"

# Match --8<-- include directives
INCLUDE_RE = re.compile(r'--8<--\s+"(\./examples/(.+?))(?::(\d+|[a-zA-Z_]\w*))?"')


def example_path_to_output_path(example_rel: str) -> str:
    """Convert an example relative path to the output file path for --8<-- inclusion.

    Example: examples/orbits/mean_motion.py -> ./docs/outputs/orbits/mean_motion.py.txt
    """
    # Strip leading "examples/" to get the category/name part
    if example_rel.startswith("examples/"):
        stripped = example_rel[len("examples/") :]
    else:
        stripped = example_rel
    return f"./docs/outputs/{stripped}.txt"


def is_line_number_offset(offset: Optional[str]) -> bool:
    """Check if an offset is a line number (digits) vs a named segment."""
    if offset is None:
        return True  # No offset means include whole file
    return offset.isdigit()


def build_output_block(
    py_output_path: str, rs_output_path: str, indent: str = ""
) -> list[str]:
    """Build the output collapsible section lines."""
    lines = [
        f"{indent}",
        f'{indent}??? example "Output"',
        f'{indent}    === "Python"',
        f"{indent}        ```",
        f'{indent}        --8<-- "{py_output_path}"',
        f"{indent}        ```",
        f"{indent}",
        f'{indent}    === "Rust"',
        f"{indent}        ```",
        f'{indent}        --8<-- "{rs_output_path}"',
        f"{indent}        ```",
    ]
    return lines


def build_single_output_block(
    output_path: str, lang: str, indent: str = ""
) -> list[str]:
    """Build a single-language output collapsible section."""
    lines = [
        f"{indent}",
        f'{indent}??? example "Output"',
        f"{indent}    ```",
        f'{indent}    --8<-- "{output_path}"',
        f"{indent}    ```",
    ]
    return lines


def process_file(file_path: Path, dry_run: bool) -> int:
    """Process a single markdown file. Returns number of output sections injected."""
    content = file_path.read_text()
    lines = content.split("\n")
    result = []
    injections = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect === "Python" tab header
        stripped = line.strip()
        if stripped == '=== "Python"':
            # Determine the indentation of the tab header
            tab_indent = line[: len(line) - len(line.lstrip())]
            content_indent = tab_indent + "    "

            # Look ahead for --8<-- inside this Python tab
            py_include = None
            py_example_rel = None
            py_offset = None
            j = i + 1

            # Scan Python tab content
            while j < len(lines):
                tab_line = lines[j]
                tab_stripped = tab_line.strip()

                # Found include directive
                m = INCLUDE_RE.search(tab_line)
                if m:
                    py_include = m.group(1)
                    py_example_rel = m.group(2)
                    py_offset = m.group(3)

                # End of Python tab: next === at same indent or blank + ===
                if tab_stripped.startswith('=== "') and tab_stripped != '=== "Python"':
                    break
                # Don't go too far looking
                if j - i > 10:
                    break
                j += 1

            # Check if we found a Python include with a line number offset
            if py_include and py_example_rel and is_line_number_offset(py_offset):
                # Now check if the next tab is === "Rust"
                rust_tab_start = None
                rs_example_rel = None

                # j should be pointing at the === "Rust" line or close to it
                k = j
                while k < len(lines) and k - j < 3:
                    if lines[k].strip() == '=== "Rust"':
                        rust_tab_start = k
                        break
                    k += 1

                if rust_tab_start is not None:
                    # Scan Rust tab content for include and closing ```
                    rust_closing_line = None
                    m_idx = rust_tab_start + 1
                    while m_idx < len(lines):
                        rust_line = lines[m_idx]
                        rust_stripped = rust_line.strip()

                        m = INCLUDE_RE.search(rust_line)
                        if m:
                            rs_example_rel = m.group(2)

                        # Closing backticks at content indent level
                        if rust_stripped == "```" and rust_line.startswith(
                            content_indent
                        ):
                            rust_closing_line = m_idx
                            break

                        # Don't scan too far
                        if m_idx - rust_tab_start > 10:
                            break
                        m_idx += 1

                    if rust_closing_line is not None and rs_example_rel:
                        # We have a complete Python/Rust tab pair
                        py_output = example_path_to_output_path(py_example_rel)
                        rs_output = example_path_to_output_path(rs_example_rel)

                        # Check if source example files exist (not output files —
                        # outputs are generated artifacts that will exist at build time)
                        py_source = REPO_ROOT / "examples" / py_example_rel
                        rs_source = REPO_ROOT / "examples" / rs_example_rel

                        # Check if output section already exists after this tab pair
                        already_has_output = False
                        look_ahead = rust_closing_line + 1
                        while (
                            look_ahead < len(lines) and lines[look_ahead].strip() == ""
                        ):
                            look_ahead += 1
                        if (
                            look_ahead < len(lines)
                            and '??? example "Output"' in lines[look_ahead]
                        ):
                            already_has_output = True

                        if not already_has_output and (
                            py_source.exists() or rs_source.exists()
                        ):
                            # Emit all lines up to and including the Rust closing ```
                            for emit_idx in range(i, rust_closing_line + 1):
                                result.append(lines[emit_idx])

                            # Inject output section — always dual-tab when both
                            # source files exist (outputs will be generated at test time)
                            if py_source.exists() and rs_source.exists():
                                output_lines = build_output_block(
                                    py_output, rs_output, tab_indent
                                )
                            elif py_source.exists():
                                output_lines = build_single_output_block(
                                    py_output, "Python", tab_indent
                                )
                            else:
                                output_lines = build_single_output_block(
                                    rs_output, "Rust", tab_indent
                                )

                            result.extend(output_lines)
                            injections += 1

                            i = rust_closing_line + 1
                            continue

        # Default: emit line as-is
        result.append(line)
        i += 1

    if injections > 0:
        new_content = "\n".join(result)
        if not dry_run:
            file_path.write_text(new_content)
            console.print(
                f"[green]Injected {injections} output section(s):[/green] "
                f"{file_path.relative_to(REPO_ROOT)}"
            )
        else:
            console.print(
                f"[yellow]Would inject {injections} output section(s):[/yellow] "
                f"{file_path.relative_to(REPO_ROOT)}"
            )

    return injections


def main(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would change without writing"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show all files processed"
    ),
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Process a single file (relative to docs/)"
    ),
):
    """Inject collapsible output sections into documentation markdown files."""
    if file:
        files = [DOCS_DIR / file]
        if not files[0].exists():
            console.print(f"[red]File not found: {files[0]}[/red]")
            raise typer.Exit(1)
    else:
        files = sorted(DOCS_DIR.glob("**/*.md"))

    total_injections = 0
    files_modified = 0

    for md_file in files:
        injections = process_file(md_file, dry_run)
        if injections > 0:
            total_injections += injections
            files_modified += 1
        elif verbose:
            console.print(f"[dim]No changes: {md_file.relative_to(REPO_ROOT)}[/dim]")

    action = "Would inject" if dry_run else "Injected"
    console.print(
        f"\n[bold]{action} {total_injections} output sections across {files_modified} files[/bold]"
    )


if __name__ == "__main__":
    typer.run(main)
