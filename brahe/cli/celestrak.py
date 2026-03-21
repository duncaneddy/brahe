"""
CLI commands for CelesTrak data queries.

Provides top-level `brahe celestrak` commands for querying GP records,
supplemental GP data, SATCAT records, and listing satellite groups.
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

import brahe as bh

from brahe.cli._output import (
    CLIOutputFormat,
    format_gp_records,
    format_satcat_records,
    parse_filters,
    CELESTRAK_SATCAT_COLUMNS,
    SATCAT_COLUMN_PRESETS,
)


class ContentFormat(str, Enum):
    tle = "tle"
    three_le = "3le"


class FileFormat(str, Enum):
    txt = "txt"
    csv = "csv"
    json = "json"


# Map supplemental GP source names to enum values
_SUP_GP_SOURCES = {
    "spacex": bh.celestrak.SupGPSource.SPACEX,
    "spacex-sup": bh.celestrak.SupGPSource.SPACEX_SUP,
    "planet": bh.celestrak.SupGPSource.PLANET,
    "oneweb": bh.celestrak.SupGPSource.ONEWEB,
    "starlink": bh.celestrak.SupGPSource.STARLINK,
    "starlink-sup": bh.celestrak.SupGPSource.STARLINK_SUP,
    "geo": bh.celestrak.SupGPSource.GEO,
    "gps": bh.celestrak.SupGPSource.GPS,
    "glonass": bh.celestrak.SupGPSource.GLONASS,
    "meteosat": bh.celestrak.SupGPSource.METEOSAT,
    "intelsat": bh.celestrak.SupGPSource.INTELSAT,
    "ses": bh.celestrak.SupGPSource.SES,
    "iridium": bh.celestrak.SupGPSource.IRIDIUM,
    "iridium-next": bh.celestrak.SupGPSource.IRIDIUM_NEXT,
    "orbcomm": bh.celestrak.SupGPSource.ORBCOMM,
    "globalstar": bh.celestrak.SupGPSource.GLOBALSTAR,
    "swarm": bh.celestrak.SupGPSource.SWARM_TECHNOLOGIES,
    "amateur": bh.celestrak.SupGPSource.AMATEUR,
    "celestrak": bh.celestrak.SupGPSource.CELESTRAK,
    "kuiper": bh.celestrak.SupGPSource.KUIPER,
}


app = typer.Typer(help="Query satellite data from CelesTrak.")


def _format_to_celestrak(content_format: str) -> "bh.celestrak.CelestrakOutputFormat":
    """Map CLI content format to CelestrakOutputFormat."""
    if content_format == "tle":
        return bh.celestrak.CelestrakOutputFormat.TLE
    elif content_format == "3le":
        return bh.celestrak.CelestrakOutputFormat.THREE_LE
    else:
        raise ValueError(f"Invalid content format: {content_format}")


@app.command()
def gp(
    group: Annotated[
        Optional[str],
        typer.Option(
            "--group",
            "-g",
            help="Satellite group name (e.g., 'stations', 'active', 'gnss')",
        ),
    ] = None,
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Satellite name search pattern"),
    ] = None,
    catnr: Annotated[
        Optional[int],
        typer.Option("--catnr", "-c", help="NORAD catalog number"),
    ] = None,
    intdes: Annotated[
        Optional[str],
        typer.Option("--intdes", help="International designator (e.g., '1998-067A')"),
    ] = None,
    filter: Annotated[
        Optional[List[str]],
        typer.Option(
            "--filter", "-f", help='Filter: "FIELD VALUE" (e.g., "INCLINATION >50")'
        ),
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", "-l", help="Maximum number of records to return"),
    ] = None,
    order_by: Annotated[
        Optional[str],
        typer.Option(
            "--order-by",
            help="Field name to sort results by (e.g., INCLINATION, EPOCH)",
        ),
    ] = None,
    descending: Annotated[
        bool,
        typer.Option("--descending", "--desc", help="Sort in descending order"),
    ] = False,
    columns: Annotated[
        Optional[str],
        typer.Option(
            "--columns",
            help="Column preset or comma-separated list (minimal, default, all)",
        ),
    ] = None,
    output_format: Annotated[
        CLIOutputFormat,
        typer.Option("--output-format", "-o", help="Output format"),
    ] = CLIOutputFormat.rich,
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output-file", help="Write output to file instead of stdout"),
    ] = None,
):
    """
    Query GP (General Perturbations) records from CelesTrak.

    At least one of --group, --name, --catnr, or --intdes must be specified.

    Examples:
        brahe celestrak gp --group stations
        brahe celestrak gp --catnr 25544
        brahe celestrak gp --group active --filter "INCLINATION >50" --limit 10
        brahe celestrak gp --group gnss --order-by INCLINATION --desc --limit 5
        brahe celestrak gp --group stations --output-format json
        brahe celestrak gp --group stations --output-format omm --output-file stations.json
        brahe celestrak gp --group stations --output-format markdown
    """
    console = Console()

    if not any([group, name, catnr, intdes]):
        console.print(
            "[red]ERROR: At least one of --group, --name, --catnr, or --intdes is required.[/red]"
        )
        raise typer.Exit(code=1)

    # Build query
    query = bh.celestrak.CelestrakQuery.gp
    if group:
        query = query.group(group)
    if name:
        query = query.name_search(name)
    if catnr is not None:
        query = query.catnr(catnr)
    if intdes:
        query = query.intdes(intdes)

    # Apply filters
    filters = parse_filters(filter)
    for field, value in filters:
        query = query.filter(field, value)

    # Apply ordering
    if order_by:
        query = query.order_by(order_by, not descending)

    # Apply limit
    if limit is not None:
        query = query.limit(limit)

    logger.info("Querying CelesTrak GP records")
    logger.debug(f"Query URL: {query.build_url()}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Querying CelesTrak...", total=None)
        try:
            client = bh.celestrak.CelestrakClient()
            records = client.query(query)
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            raise typer.Exit(code=1)

    format_gp_records(records, output_format, columns, output_file, console)


@app.command("sup-gp")
def sup_gp(
    source: Annotated[
        str,
        typer.Argument(
            help="Supplemental GP source (e.g., starlink, planet, oneweb, spacex, gps, glonass)"
        ),
    ],
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
        typer.Option("--order-by", help="Field name to sort results by"),
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
    Query supplemental GP records from CelesTrak.

    Supplemental GP data provides operator-provided ephemerides that may be
    more accurate than standard GP data for certain constellations.

    Available sources: spacex, planet, oneweb, starlink, geo, gps, glonass,
    meteosat, intelsat, ses, iridium, orbcomm, globalstar, swarm, amateur,
    celestrak, kuiper

    Examples:
        brahe celestrak sup-gp starlink --limit 10
        brahe celestrak sup-gp planet --output-format json
        brahe celestrak sup-gp oneweb --filter "INCLINATION >80"
    """
    console = Console()

    source_lower = source.lower()
    if source_lower not in _SUP_GP_SOURCES:
        console.print(f"[red]ERROR: Unknown source '{source}'.[/red]")
        console.print(
            f"[yellow]Available sources: {', '.join(sorted(_SUP_GP_SOURCES.keys()))}[/yellow]"
        )
        raise typer.Exit(code=1)

    sup_source = _SUP_GP_SOURCES[source_lower]
    query = bh.celestrak.CelestrakQuery.sup_gp.source(sup_source)

    filters = parse_filters(filter)
    for field, value in filters:
        query = query.filter(field, value)
    if order_by:
        query = query.order_by(order_by, not descending)
    if limit is not None:
        query = query.limit(limit)

    logger.info(f"Querying CelesTrak supplemental GP for {source}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description=f"Querying CelesTrak sup-gp ({source})...", total=None
        )
        try:
            client = bh.celestrak.CelestrakClient()
            records = client.query(query)
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
    active: Annotated[
        Optional[bool],
        typer.Option("--active", help="Filter to active objects only"),
    ] = None,
    payloads: Annotated[
        Optional[bool],
        typer.Option("--payloads", help="Filter to payloads only"),
    ] = None,
    on_orbit: Annotated[
        Optional[bool],
        typer.Option("--on-orbit", help="Filter to on-orbit objects only"),
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
    Query SATCAT (Satellite Catalog) records from CelesTrak.

    At least one filter option must be specified.

    Examples:
        brahe celestrak satcat --catnr 25544
        brahe celestrak satcat --active --payloads --output-format json
        brahe celestrak satcat --on-orbit --output-format markdown
    """
    console = Console()

    if not any(
        [
            catnr is not None,
            active is not None,
            payloads is not None,
            on_orbit is not None,
        ]
    ):
        console.print(
            "[red]ERROR: At least one of --catnr, --active, --payloads, or --on-orbit is required.[/red]"
        )
        raise typer.Exit(code=1)

    query = bh.celestrak.CelestrakQuery.satcat
    if catnr is not None:
        query = query.catnr(catnr)
    if active is not None:
        query = query.active(active)
    if payloads is not None:
        query = query.payloads(payloads)
    if on_orbit is not None:
        query = query.on_orbit(on_orbit)

    logger.info("Querying CelesTrak SATCAT")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Querying CelesTrak SATCAT...", total=None)
        try:
            client = bh.celestrak.CelestrakClient()
            records = client.query_satcat(query)
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            raise typer.Exit(code=1)

    format_satcat_records(
        records,
        output_format,
        columns,
        output_file,
        console,
        columns_def=CELESTRAK_SATCAT_COLUMNS,
        presets=SATCAT_COLUMN_PRESETS,
    )


@app.command()
def groups():
    """
    List commonly used CelesTrak satellite groups.

    Examples:
        brahe celestrak groups
    """
    console = Console()

    group_list = [
        ("active", "All active satellites"),
        ("stations", "Space stations (ISS, Tiangong, etc.)"),
        ("last-30-days", "Satellites launched in the last 30 days"),
        ("gnss", "All GNSS satellites (GPS, Galileo, GLONASS, Beidou)"),
        ("gps-ops", "Operational GPS satellites"),
        ("galileo", "Galileo navigation satellites"),
        ("beidou", "Beidou navigation satellites"),
        ("glo-ops", "Operational GLONASS satellites"),
        ("geo", "Geostationary satellites"),
        ("gpz", "Geostationary protected zone satellites"),
        ("gpz-plus", "Geostationary protected zone plus satellites"),
        ("weather", "Weather satellites"),
        ("noaa", "NOAA satellites"),
        ("goes", "GOES weather satellites"),
        ("starlink", "SpaceX Starlink constellation"),
        ("oneweb", "OneWeb constellation"),
        ("kuiper", "Amazon Kuiper constellation"),
        ("qianfan", "Qianfan constellation"),
        ("hulianwang", "Hulianwang constellation"),
        ("planet", "Planet Labs imaging satellites"),
        ("iridium", "Iridium constellation"),
        ("iridium-NEXT", "Iridium NEXT constellation"),
        ("intelsat", "Intelsat satellites"),
        ("eutelsat", "Eutelsat satellites"),
        ("ses", "SES satellites"),
        ("orbcomm", "Orbcomm satellites"),
        ("globalstar", "Globalstar satellites"),
        ("sarsat", "Search and rescue satellites"),
        ("cubesat", "CubeSats"),
        ("amateur", "Amateur radio satellites"),
        ("science", "Science satellites"),
        ("geodetic", "Geodetic satellites"),
        ("cosmos-2251-debris", "Debris from Cosmos 2251 collision"),
        ("iridium-33-debris", "Debris from Iridium 33 collision"),
        ("fengyun-1c-debris", "Debris from Fengyun-1C ASAT test"),
        ("cosmos-1408-debris", "Debris from Cosmos 1408 ASAT test"),
    ]

    console.print("\n[bold cyan]CelesTrak Satellite Groups[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Group Name", style="cyan", width=20)
    table.add_column("Description", style="white")

    for group_name, description in group_list:
        table.add_row(group_name, description)

    console.print(table)
    console.print(
        f"\n[dim]{len(group_list)} groups shown. "
        f"For complete list, visit https://celestrak.org/NORAD/elements/[/dim]\n"
    )


@app.command()
def download(
    filepath: Annotated[Path, typer.Argument(help="Output file path")],
    group: Annotated[
        str,
        typer.Option("--group", "-g", help="Satellite group name"),
    ] = "active",
    content_format: Annotated[
        ContentFormat,
        typer.Option(
            "--content-format",
            help="Content format: 'tle' (2-line) or '3le' (3-line with names)",
        ),
    ] = ContentFormat.three_le,
    file_format: Annotated[
        FileFormat,
        typer.Option("--file-format", help="File format: 'txt', 'csv', or 'json'"),
    ] = FileFormat.txt,
):
    """
    Download satellite ephemeris data from CelesTrak and save to file.

    Examples:
        brahe celestrak download data.json --group gnss --content-format 3le --file-format json
        brahe celestrak download sats.txt --group active --content-format tle --file-format txt
    """
    logger.info(f"Downloading CelesTrak {group} group to {filepath}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description=f"Downloading {group} satellites from CelesTrak...", total=None
        )
        try:
            client = bh.celestrak.CelestrakClient()
            fmt = _format_to_celestrak(content_format.value)
            query = bh.celestrak.CelestrakQuery.gp.group(group).format(fmt)
            client.download(query, str(filepath.absolute()))
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    typer.echo(f"Downloaded {group} satellites to {filepath}")
