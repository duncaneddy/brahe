"""CLI commands for ICGEM gravity model datasets."""

from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

import brahe.datasets as datasets


app = typer.Typer()


_KNOWN_BODIES = ["earth", "moon", "mars", "venus", "ceres"]


@app.command("list")
def list_models(
    body: Annotated[
        str, typer.Option("--body", "-b", help="Body name or 'all'.")
    ] = "earth",
    table_view: Annotated[
        bool, typer.Option("--table", "-t", help="Display as a Rich table.")
    ] = False,
):
    """List ICGEM gravity models for a body (or all bodies)."""
    multi_body = body.lower() == "all"
    bodies = _KNOWN_BODIES if multi_body else [body]

    rows = []
    for b in bodies:
        try:
            entries = datasets.icgem.list_models(b)
        except Exception as e:
            # In --body all mode, an individual body's failure shouldn't doom
            # the whole listing — log and continue. In single-body mode, the
            # user asked about THAT body specifically, so propagate the error
            # rather than silently returning an empty list.
            if multi_body:
                logger.warning(f"Skipping {b}: {e}")
                continue
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1) from e
        for e in entries:
            rows.append((e.body, e.name, e.degree, e.year))

    if not table_view:
        for body_, name, degree, year in rows:
            typer.echo(f"{body_:8s} {name:40s} degree={degree:<6d} year={year}")
        typer.echo(f"\n{len(rows)} model(s)")
        return

    console = Console()
    rich_table = Table(show_header=True, header_style="bold cyan")
    rich_table.add_column("Body", style="cyan")
    rich_table.add_column("Model", style="green")
    rich_table.add_column("Degree", justify="right", style="yellow")
    rich_table.add_column("Year", justify="right")
    for body_, name, degree, year in rows:
        rich_table.add_row(body_, name, str(degree), str(year if year else ""))
    console.print(rich_table)


@app.command("download")
def download(
    name: Annotated[str, typer.Argument(help="Model name (optionally NAME-DEGREE).")],
    body: Annotated[str, typer.Option("--body", "-b", help="Body name.")] = "earth",
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Copy the file to this path."),
    ] = None,
):
    """Download an ICGEM gravity model into the local cache."""
    if body.lower() == "all":
        raise typer.BadParameter("`--body all` is not valid for `download`")
    try:
        path = datasets.icgem.download_model(body, name, output)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(path)


@app.command("refresh")
def refresh(
    body: Annotated[
        Optional[str], typer.Option("--body", "-b", help="Body to refresh.")
    ] = None,
    all_bodies: Annotated[
        bool, typer.Option("--all", help="Refresh both Earth and celestial indexes.")
    ] = False,
):
    """Force-refresh the ICGEM index files."""
    if body and all_bodies:
        raise typer.BadParameter("Pass either `--body` or `--all`, not both.")
    if not body and not all_bodies:
        raise typer.BadParameter("Pass either `--body` or `--all`.")

    if all_bodies or (body and body.lower() == "all"):
        datasets.icgem.refresh_all_indexes()
        typer.echo("Refreshed all ICGEM indexes.")
    else:
        datasets.icgem.refresh_index(body)
        typer.echo(f"Refreshed ICGEM index for body '{body}'.")
