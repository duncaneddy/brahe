#!/usr/bin/env python3
"""Build packages for distribution (dry-run validation)."""

import subprocess
import tempfile
from pathlib import Path

import typer

from _build_utils import REPO_ROOT, console


def main(
    lang: str = typer.Option("both", "--lang", help="Language: python, rust, or both"),
    release: bool = typer.Option(False, "--release", help="Build in release mode"),
):
    """
    Build packages for distribution (dry-run validation).

    Validates that packages build correctly and meet publishing requirements:
    - Rust: Checks package size < 10MB compressed, builds successfully
    - Python: Checks PKG-INFO metadata is correct, passes twine validation
    """
    console.print("\n[bold]Building Packages[/bold]\n")

    if lang not in ["python", "rust", "both"]:
        console.print("[red]Error: --lang must be 'python', 'rust', or 'both'[/red]")
        raise typer.Exit(1)

    build_rust = lang in ["rust", "both"]
    build_python = lang in ["python", "both"]

    # Rust build
    if build_rust:
        console.print("[bold cyan]Building Rust crate...[/bold cyan]")

        result = subprocess.run(
            ["cargo", "package", "--no-verify", "--allow-dirty"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            console.print("[red]✗ Cargo package failed[/red]")
            console.print(result.stderr)
            raise typer.Exit(1)

        crate_files = list((REPO_ROOT / "target" / "package").glob("*.crate"))
        if not crate_files:
            console.print("[red]✗ No .crate file found[/red]")
            raise typer.Exit(1)

        crate_file = crate_files[0]
        size_bytes = crate_file.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        console.print(f"Package size: {size_mb:.2f}MB compressed")

        if size_bytes > 10 * 1024 * 1024:
            console.print(
                f"[red]✗ Package size ({size_mb:.2f}MB) exceeds 10MB limit[/red]"
            )
            raise typer.Exit(1)

        console.print("[green]✓ Package size OK[/green]")

        console.print("Testing packaged crate builds...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            subprocess.run(
                ["tar", "-xzf", str(crate_file)],
                cwd=tmpdir_path,
                check=True,
                capture_output=True,
            )

            extracted_dirs = [d for d in tmpdir_path.iterdir() if d.is_dir()]
            if not extracted_dirs:
                console.print("[red]✗ Failed to extract crate[/red]")
                raise typer.Exit(1)

            build_args = ["cargo", "build"]
            if release:
                build_args.append("--release")

            result = subprocess.run(
                build_args, cwd=extracted_dirs[0], capture_output=True, text=True
            )

            if result.returncode != 0:
                console.print("[red]✗ Packaged crate failed to build[/red]")
                console.print(result.stderr)
                raise typer.Exit(1)

        console.print("[green]✓ Packaged crate builds successfully[/green]\n")

    # Python build
    if build_python:
        console.print("[bold cyan]Building Python packages...[/bold cyan]")

        console.print("Building source distribution...")
        result = subprocess.run(
            ["uv", "run", "maturin", "build", "--sdist"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            console.print("[red]✗ Maturin sdist build failed[/red]")
            console.print(result.stderr)
            raise typer.Exit(1)

        console.print("[green]✓ Source distribution built[/green]")

        console.print("Building wheel...")
        wheel_args = ["uv", "run", "maturin", "build"]
        if release:
            wheel_args.append("--release")

        result = subprocess.run(
            wheel_args, cwd=REPO_ROOT, capture_output=True, text=True
        )

        if result.returncode != 0:
            console.print("[red]✗ Maturin wheel build failed[/red]")
            console.print(result.stderr)
            raise typer.Exit(1)

        console.print("[green]✓ Wheel built[/green]")

        # Validate PKG-INFO metadata
        console.print("Validating PKG-INFO metadata...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            sdist_files = list((REPO_ROOT / "target" / "wheels").glob("*.tar.gz"))
            if not sdist_files:
                console.print("[yellow]⚠ No sdist found to validate[/yellow]")
            else:
                subprocess.run(
                    ["tar", "-xzf", str(sdist_files[0])],
                    cwd=tmpdir_path,
                    check=True,
                    capture_output=True,
                )

                pkg_info_files = list(tmpdir_path.glob("**/PKG-INFO"))
                if not pkg_info_files:
                    console.print("[red]✗ No PKG-INFO found in sdist[/red]")
                    raise typer.Exit(1)

                pkg_info_content = pkg_info_files[0].read_text()

                if "Description-Content-Type: text/markdown" not in pkg_info_content:
                    console.print(
                        "[red]✗ Missing or incorrect Description-Content-Type[/red]"
                    )
                    console.print("PKG-INFO contents:")
                    console.print(pkg_info_content)
                    raise typer.Exit(1)

                console.print(
                    "[green]✓ Description-Content-Type: text/markdown found[/green]"
                )

                lines = pkg_info_content.split("\n")
                summary_idx = next(
                    (i for i, line in enumerate(lines) if line.startswith("Summary:")),
                    None,
                )
                if summary_idx is not None and summary_idx + 1 < len(lines):
                    if lines[summary_idx + 1].strip() == "":
                        console.print(
                            "[red]✗ Summary field appears to be multi-line[/red]"
                        )
                        raise typer.Exit(1)

                console.print("[green]✓ Summary field is properly formatted[/green]")

        # Validate with twine
        console.print("Running twine check...")

        wheel_files = list((REPO_ROOT / "target" / "wheels").glob("*.whl"))
        sdist_files = list((REPO_ROOT / "target" / "wheels").glob("*.tar.gz"))
        all_files = [str(f) for f in wheel_files + sdist_files]

        if not all_files:
            console.print("[yellow]⚠ No packages found to validate with twine[/yellow]")
        else:
            result = subprocess.run(
                ["uv", "run", "twine", "check"] + all_files,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                console.print("[red]✗ Twine validation failed[/red]")
                console.print(result.stdout)
                console.print(result.stderr)
                raise typer.Exit(1)

            console.print("[green]✓ All packages pass twine validation[/green]\n")

    console.print("[bold green]✓ All builds successful![/bold green]\n")


if __name__ == "__main__":
    typer.run(main)
