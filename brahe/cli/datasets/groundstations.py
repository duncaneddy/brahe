"""
CLI commands for groundstation datasets
"""

from typing import Optional
import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.table import Table
import brahe.datasets as datasets


app = typer.Typer()


@app.command("list-providers")
def list_providers(
    table: Annotated[
        bool, typer.Option("--table", "-t", help="Display as rich table")
    ] = False,
):
    """
    List available groundstation providers.

    Examples:
        brahe datasets groundstations list-providers
        brahe datasets groundstations list-providers -t
    """
    providers = datasets.groundstations.list_providers()

    if not table:
        # Simple text output
        typer.echo("Available groundstation providers:")
        for provider in providers:
            typer.echo(f"  - {provider}")
        return

    # Rich table output - gather metadata for each provider
    console = Console()
    provider_data = []

    for provider in providers:
        try:
            stations = datasets.groundstations.load(provider)
            station_count = len(stations)

            # Collect unique frequency bands across all stations
            bands = set()
            for station in stations:
                if "frequency_bands" in station.properties:
                    bands.update(station.properties["frequency_bands"])

            bands_str = ", ".join(sorted(bands)) if bands else "N/A"
            provider_data.append((provider, station_count, bands_str))
        except Exception:
            # If provider fails to load, show with N/A
            provider_data.append((provider, "Error", "N/A"))

    # Create rich table
    rich_table = Table(show_header=True, header_style="bold cyan")
    rich_table.add_column("Provider", style="cyan")
    rich_table.add_column("Stations", justify="right", style="yellow")
    rich_table.add_column("Frequency Bands", style="green")

    for provider, count, bands in provider_data:
        rich_table.add_row(provider, str(count), bands)

    console.print()
    console.print(rich_table)
    console.print()


@app.command("list-stations")
def list_stations(
    provider: Annotated[
        Optional[str],
        typer.Option(help="Filter by provider (e.g., 'ksat', 'atlas', 'aws')"),
    ] = None,
    table: Annotated[
        bool, typer.Option("--table", "-t", help="Display as rich table")
    ] = False,
):
    """
    List groundstations, optionally filtered by provider.

    Examples:
        brahe datasets groundstations list-stations
        brahe datasets groundstations list-stations --provider ksat
        brahe datasets groundstations list-stations --provider ksat -t
        brahe datasets groundstations list-stations -t
    """
    # Load stations
    try:
        if provider:
            stations = datasets.groundstations.load(provider)
        else:
            stations = datasets.groundstations.load_all()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    if not stations:
        typer.echo("No groundstations found.")
        return

    if not table:
        # Simple text output
        if provider:
            typer.echo(f"\n{provider.upper()} Groundstations ({len(stations)} total):")
        else:
            typer.echo(f"\nAll Groundstations ({len(stations)} total):")

        for station in stations:
            name = station.get_name() if station.get_name() else "Unnamed"
            # Coordinates are already in degrees
            lon = station.lon
            lat = station.lat
            alt = station.alt

            props = station.properties
            bands = (
                ", ".join(props.get("frequency_bands", []))
                if "frequency_bands" in props
                else "N/A"
            )

            typer.echo(
                f"  {name}: {lat:.3f}° lat, {lon:.3f}° lon, {alt:.0f} m alt [{bands}]"
            )

        typer.echo(f"\n✓ Listed {len(stations)} groundstation(s)")
        return

    # Rich table output
    console = Console()
    rich_table = Table(show_header=True, header_style="bold cyan")
    rich_table.add_column("Name", style="cyan", width=25)
    rich_table.add_column("Lat (°)", justify="right", style="yellow")
    rich_table.add_column("Lon (°)", justify="right", style="yellow")
    rich_table.add_column("Alt (m)", justify="right", style="yellow")
    rich_table.add_column("Provider", style="green")
    rich_table.add_column("Frequency Bands", style="magenta")

    for station in stations:
        name = station.get_name() if station.get_name() else "Unnamed"
        # Coordinates are already in degrees
        lon = station.lon
        lat = station.lat
        alt = station.alt

        props = station.properties
        provider_name = props.get("provider", "N/A")
        bands = (
            ", ".join(props.get("frequency_bands", []))
            if "frequency_bands" in props
            else "N/A"
        )

        rich_table.add_row(
            name,
            f"{lat:.3f}",
            f"{lon:.3f}",
            f"{alt:.0f}",
            provider_name,
            bands,
        )

    console.print()
    if provider:
        console.print(
            f"[bold]{provider.upper()} Groundstations ({len(stations)} total)[/bold]"
        )
    else:
        console.print(f"[bold]All Groundstations ({len(stations)} total)[/bold]")
    console.print(rich_table)
    console.print()


@app.command()
def show(
    provider: Annotated[
        str, typer.Argument(help="Provider name (e.g., 'ksat', 'atlas', 'aws')")
    ],
    show_properties: Annotated[
        bool, typer.Option("--properties", "-p", help="Show station properties")
    ] = False,
):
    """
    Show groundstations for a specific provider (deprecated - use list-stations).

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
        name = station.get_name() if station.get_name() else "Unnamed"
        # Coordinates are already in degrees
        lon = station.lon
        lat = station.lat
        alt = station.alt

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


@app.command("show-all")
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
            lon = station.lon
            lat = station.lat
            alt = station.alt

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
