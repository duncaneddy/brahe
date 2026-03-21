"""
CLI commands for SpaceTrack data queries.

Provides top-level `brahe spacetrack` commands for querying GP records,
GP history, and SATCAT records from Space-Track.org.

Requires SPACETRACK_USER and SPACETRACK_PASS environment variables.
"""

import os
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

import brahe as bh
from brahe.spacetrack import operators as op

from brahe.cli._output import (
    CLIOutputFormat,
    format_gp_records,
    format_satcat_records,
    parse_filters,
)


app = typer.Typer(help="Query satellite data from Space-Track.org.")


def _get_client(console: Console) -> bh.SpaceTrackClient:
    """Create SpaceTrack client from environment variables."""
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")
    if not user or not password:
        console.print(
            "[red]ERROR: SPACETRACK_USER and SPACETRACK_PASS environment variables must be set.[/red]\n"
            "[dim]Sign up at https://www.space-track.org/auth/createAccount[/dim]"
        )
        raise typer.Exit(code=1)
    return bh.SpaceTrackClient(user, password)


def _apply_common_options(query, filters, order_by, descending, limit):
    """Apply common filter/sort/limit options to a SpaceTrackQuery."""
    for field, value in filters:
        query = query.filter(field, value)
    if order_by:
        sort_order = bh.SortOrder.DESC if descending else bh.SortOrder.ASC
        query = query.order_by(order_by, sort_order)
    if limit is not None:
        query = query.limit(limit)
    return query


@app.command()
def gp(
    catnr: Annotated[
        Optional[int],
        typer.Option("--catnr", "-c", help="NORAD catalog number"),
    ] = None,
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Satellite name search pattern"),
    ] = None,
    epoch_range: Annotated[
        Optional[str],
        typer.Option(
            "--epoch-range", help="Epoch range filter 'START--END' (ISO-8601)"
        ),
    ] = None,
    filter: Annotated[
        Optional[List[str]],
        typer.Option(
            "--filter", "-f", help='Filter: "FIELD VALUE" (e.g., "INCLINATION >50")'
        ),
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", "-l", help="Maximum number of records"),
    ] = None,
    order_by: Annotated[
        Optional[str],
        typer.Option(
            "--order-by", help="Field name to sort by (e.g., EPOCH, INCLINATION)"
        ),
    ] = None,
    descending: Annotated[
        bool,
        typer.Option("--descending", "--desc", help="Sort in descending order"),
    ] = False,
    columns: Annotated[
        Optional[str],
        typer.Option("--columns", help="Column preset or comma-separated list"),
    ] = None,
    output_format: Annotated[
        CLIOutputFormat,
        typer.Option("--output-format", "-o", help="Output format"),
    ] = CLIOutputFormat.rich,
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output-file", help="Write output to file"),
    ] = None,
):
    """
    Query GP (General Perturbations) records from Space-Track.org.

    At least one of --catnr, --name, or --filter must be specified.

    Examples:
        brahe spacetrack gp --catnr 25544
        brahe spacetrack gp --name "ISS" --limit 5
        brahe spacetrack gp --filter "NORAD_CAT_ID 25544" --output-format json
        brahe spacetrack gp --catnr 25544 --output-format omm
        brahe spacetrack gp --filter "INCLINATION >50" --limit 10 --order-by INCLINATION
    """
    console = Console()

    parsed_filters = parse_filters(filter)

    if not any([catnr is not None, name, parsed_filters]):
        console.print(
            "[red]ERROR: At least one of --catnr, --name, or --filter is required.[/red]"
        )
        raise typer.Exit(code=1)

    client = _get_client(console)

    query = bh.SpaceTrackQuery(bh.RequestClass.GP)

    # Apply named filters
    if catnr is not None:
        query = query.filter("NORAD_CAT_ID", str(catnr))
    if name:
        query = query.filter("OBJECT_NAME", op.like(name))
    if epoch_range:
        query = query.filter("EPOCH", epoch_range)

    query = _apply_common_options(query, parsed_filters, order_by, descending, limit)

    logger.info("Querying SpaceTrack GP records")
    logger.debug(f"Query: {query}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Querying Space-Track.org...", total=None)
        try:
            records = client.query_gp(query)
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            raise typer.Exit(code=1)

    format_gp_records(records, output_format, columns, output_file, console)


@app.command("gp-history")
def gp_history(
    catnr: Annotated[
        int,
        typer.Argument(help="NORAD catalog number"),
    ],
    epoch_range: Annotated[
        Optional[str],
        typer.Option(
            "--epoch-range", help="Epoch range filter 'START--END' (ISO-8601)"
        ),
    ] = None,
    filter: Annotated[
        Optional[List[str]],
        typer.Option("--filter", "-f", help='Filter: "FIELD VALUE"'),
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", "-l", help="Maximum number of records"),
    ] = None,
    order_by: Annotated[
        Optional[str],
        typer.Option("--order-by", help="Field name to sort by"),
    ] = None,
    descending: Annotated[
        bool,
        typer.Option("--descending", "--desc", help="Sort in descending order"),
    ] = False,
    columns: Annotated[
        Optional[str],
        typer.Option("--columns", help="Column preset or comma-separated list"),
    ] = None,
    output_format: Annotated[
        CLIOutputFormat,
        typer.Option("--output-format", "-o", help="Output format"),
    ] = CLIOutputFormat.rich,
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output-file", help="Write output to file"),
    ] = None,
):
    """
    Query GP history records from Space-Track.org for a specific satellite.

    Returns historical element sets showing orbital evolution over time.
    An --epoch-range is recommended to avoid very large result sets.

    Examples:
        brahe spacetrack gp-history 25544 --limit 10
        brahe spacetrack gp-history 25544 --epoch-range "2024-01-01--2024-01-31"
        brahe spacetrack gp-history 25544 --order-by EPOCH --desc --limit 5
        brahe spacetrack gp-history 25544 --output-format json --output-file iss_history.json
    """
    console = Console()
    client = _get_client(console)

    query = bh.SpaceTrackQuery(bh.RequestClass.GP_HISTORY)
    query = query.filter("NORAD_CAT_ID", str(catnr))

    if epoch_range:
        query = query.filter("EPOCH", epoch_range)

    parsed_filters = parse_filters(filter)
    query = _apply_common_options(query, parsed_filters, order_by, descending, limit)

    logger.info(f"Querying SpaceTrack GP history for NORAD {catnr}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Querying GP history for {catnr}...", total=None)
        try:
            records = client.query_gp(query)
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            raise typer.Exit(code=1)

    format_gp_records(records, output_format, columns, output_file, console)


@app.command()
def satcat(
    catnr: Annotated[
        Optional[int],
        typer.Option("--catnr", "-c", help="NORAD catalog number"),
    ] = None,
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Satellite name search pattern"),
    ] = None,
    country: Annotated[
        Optional[str],
        typer.Option("--country", help="Country code filter (e.g., US, CIS, CN)"),
    ] = None,
    object_type: Annotated[
        Optional[str],
        typer.Option(
            "--object-type", help="Object type filter (PAYLOAD, ROCKET BODY, DEBRIS)"
        ),
    ] = None,
    filter: Annotated[
        Optional[List[str]],
        typer.Option("--filter", "-f", help='Filter: "FIELD VALUE"'),
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", "-l", help="Maximum number of records"),
    ] = None,
    columns: Annotated[
        Optional[str],
        typer.Option("--columns", help="Column preset or comma-separated list"),
    ] = None,
    output_format: Annotated[
        CLIOutputFormat,
        typer.Option("--output-format", "-o", help="Output format"),
    ] = CLIOutputFormat.rich,
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output-file", help="Write output to file"),
    ] = None,
):
    """
    Query SATCAT (Satellite Catalog) records from Space-Track.org.

    At least one of --catnr, --name, --country, --object-type, or --filter must be specified.

    Examples:
        brahe spacetrack satcat --catnr 25544
        brahe spacetrack satcat --name "ISS" --limit 5
        brahe spacetrack satcat --country US --object-type PAYLOAD --limit 20
        brahe spacetrack satcat --filter "COUNTRY US" --output-format json
    """
    console = Console()

    parsed_filters = parse_filters(filter)

    if not any([catnr is not None, name, country, object_type, parsed_filters]):
        console.print(
            "[red]ERROR: At least one of --catnr, --name, --country, --object-type, or --filter is required.[/red]"
        )
        raise typer.Exit(code=1)

    client = _get_client(console)

    query = bh.SpaceTrackQuery(bh.RequestClass.SATCAT)

    if catnr is not None:
        query = query.filter("NORAD_CAT_ID", str(catnr))
    if name:
        query = query.filter("SATNAME", op.like(name))
    if country:
        query = query.filter("COUNTRY", country)
    if object_type:
        query = query.filter("OBJECT_TYPE", object_type)

    for field, value in parsed_filters:
        query = query.filter(field, value)
    if limit is not None:
        query = query.limit(limit)

    logger.info("Querying SpaceTrack SATCAT")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Querying SpaceTrack SATCAT...", total=None)
        try:
            records = client.query_satcat(query)
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            raise typer.Exit(code=1)

    format_satcat_records(records, output_format, columns, output_file, console)
