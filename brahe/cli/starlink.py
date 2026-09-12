"""Starlink public ephemeris commands."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

import brahe as bh

app = typer.Typer(help="Retrieve Starlink public ephemerides.")
console = Console()


@app.command()
def manifest(
    norad_id: Annotated[
        int | None,
        typer.Option("--norad-id", help="Show only this NORAD catalog number."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Show only this object name (exact match)."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Show at most this many entries.")
    ] = None,
) -> None:
    """List the satellites in the Starlink manifest."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Fetching manifest...", total=None)
            client = bh.starlink.StarlinkClient()
            listing = client.get_manifest()
    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        raise typer.Exit(code=1) from e

    if norad_id is not None:
        entry = listing.find_by_norad_id(norad_id)
        if entry is None:
            console.print(
                f"[red]ERROR: NORAD ID {norad_id} is not in the manifest[/red]"
            )
            raise typer.Exit(code=1)
        entries = [entry]
    elif name is not None:
        entry = listing.find_by_object_name(name)
        if entry is None:
            console.print(f"[red]ERROR: object {name} is not in the manifest[/red]")
            raise typer.Exit(code=1)
        entries = [entry]
    else:
        entries = list(listing.entries())
    total = len(listing)
    if limit is not None:
        entries = entries[:limit]

    table = Table(title=f"Starlink manifest ({len(entries)} of {total} entries)")
    table.add_column("NORAD ID", justify="right")
    table.add_column("Object")
    table.add_column("Category")
    table.add_column("Ephemeris start")
    table.add_column("Ephemeris stop")
    table.add_column("File")
    for entry in entries:
        table.add_row(
            str(entry.norad_cat_id),
            entry.object_name,
            str(entry.category),
            str(entry.ephemeris_start),
            str(entry.ephemeris_stop) if entry.ephemeris_stop is not None else "",
            entry.file_name_string(),
        )
    console.print(table)


@app.command()
def download(
    norad_ids: Annotated[
        list[int], typer.Argument(help="NORAD catalog numbers to download.")
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Directory to copy the files into; the cache is used when omitted.",
        ),
    ] = None,
) -> None:
    """Download ephemeris files for one or more satellites."""
    client = bh.starlink.StarlinkClient()
    for norad_id in norad_ids:
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(f"Downloading {norad_id}...", total=None)
                if output is None:
                    path = client.download_ephemeris(norad_id)
                else:
                    path = client.save_ephemeris(norad_id, str(output))
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            raise typer.Exit(code=1) from e
        console.print(f"Downloaded {norad_id} to {path}")
