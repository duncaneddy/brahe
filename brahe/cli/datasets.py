"""
CLI commands for datasets module
"""

from enum import Enum
from pathlib import Path
import typer
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn
import brahe.datasets as datasets


class ContentFormat(str, Enum):
    tle = "tle"
    three_le = "3le"


class FileFormat(str, Enum):
    txt = "txt"
    csv = "csv"
    json = "json"


app = typer.Typer()
celestrak_app = typer.Typer()
groundstations_app = typer.Typer()


@celestrak_app.command()
def download(
    filepath: Annotated[Path, typer.Argument(help="Output file path")],
    group: Annotated[
        str,
        typer.Option(help="Satellite group name (e.g., 'active', 'stations', 'gnss')"),
    ] = "active",
    content_format: Annotated[
        ContentFormat,
        typer.Option(
            help="Content format: 'tle' (2-line) or '3le' (3-line with names)"
        ),
    ] = ContentFormat.three_le,
    file_format: Annotated[
        FileFormat, typer.Option(help="File format: 'txt', 'csv', or 'json'")
    ] = FileFormat.txt,
):
    """
    Download satellite ephemeris data from CelesTrak and save to file.

    Examples:
        brahe datasets celestrak download data.json --group gnss --content-format 3le --file-format json
        brahe datasets celestrak download sats.txt --group active --content-format tle --file-format txt
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description=f"Downloading {group} satellites from CelesTrak...", total=None
        )

        try:
            datasets.celestrak.download_ephemeris(
                group, str(filepath.absolute()), content_format.value, file_format.value
            )
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    typer.echo(f"✓ Downloaded {group} satellites to {filepath}")


@groundstations_app.command()
def list():
    """
    List available groundstation providers.

    Examples:
        brahe datasets groundstations list
    """
    providers = datasets.groundstations.list_providers()

    typer.echo("Available groundstation providers:")
    for provider in providers:
        typer.echo(f"  - {provider}")


@groundstations_app.command()
def show(
    provider: Annotated[
        str, typer.Argument(help="Provider name (e.g., 'ksat', 'atlas', 'aws')")
    ],
    show_properties: Annotated[
        bool, typer.Option("--properties", "-p", help="Show station properties")
    ] = False,
):
    """
    Show groundstations for a specific provider.

    Examples:
        brahe datasets groundstations show ksat
        brahe datasets groundstations show atlas --properties
    """
    try:
        stations = datasets.groundstations.load(provider)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\n{provider.upper()} Groundstations ({len(stations)} total):")
    typer.echo("-" * 80)

    for station in stations:
        name = station.name if hasattr(station, "name") and station.name else "Unnamed"
        lon = station.lon()
        lat = station.lat()
        alt = station.alt()

        typer.echo(f"\n{name}")
        typer.echo(f"  Location: {lon:8.3f}° lon, {lat:7.3f}° lat, {alt:6.0f} m alt")

        if show_properties:
            props = station.properties
            if "frequency_bands" in props:
                bands = ", ".join(props["frequency_bands"])
                typer.echo(f"  Frequency bands: {bands}")
            if "provider" in props:
                typer.echo(f"  Provider: {props['provider']}")

    typer.echo(f"\n✓ Loaded {len(stations)} groundstations from {provider}")


@groundstations_app.command("show-all")
def show_all(
    show_properties: Annotated[
        bool, typer.Option("--properties", "-p", help="Show station properties")
    ] = False,
):
    """
    Show groundstations from all providers.

    Examples:
        brahe datasets groundstations show-all
        brahe datasets groundstations show-all --properties
    """
    try:
        all_stations = datasets.groundstations.load_all()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    # Group by provider
    by_provider = {}
    for station in all_stations:
        props = station.properties
        provider = props.get("provider", "Unknown")
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(station)

    typer.echo(f"\nAll Groundstations ({len(all_stations)} total):")
    typer.echo("=" * 80)

    for provider, stations in sorted(by_provider.items()):
        typer.echo(f"\n{provider} ({len(stations)} stations):")
        typer.echo("-" * 80)

        for station in stations:
            name = (
                station.name if hasattr(station, "name") and station.name else "Unnamed"
            )
            lon = station.lon()
            lat = station.lat()
            alt = station.alt()

            typer.echo(f"\n  {name}")
            typer.echo(
                f"    Location: {lon:8.3f}° lon, {lat:7.3f}° lat, {alt:6.0f} m alt"
            )

            if show_properties:
                props = station.properties
                if "frequency_bands" in props:
                    bands = ", ".join(props["frequency_bands"])
                    typer.echo(f"    Frequency bands: {bands}")

    typer.echo(f"\n✓ Loaded {len(all_stations)} groundstations from all providers")


app.add_typer(celestrak_app, name="celestrak")
app.add_typer(groundstations_app, name="groundstations")
