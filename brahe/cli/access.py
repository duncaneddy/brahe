import json
import math
from enum import Enum
from pathlib import Path
from typing import Optional
import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.table import Table

import brahe
from brahe.cli.utils import set_cli_eop

app = typer.Typer()


class OutputFormat(str, Enum):
    """Output format for access window display."""

    rich = "rich"
    simple = "simple"


@app.command()
def compute(
    norad_id: Annotated[int, typer.Argument(help="NORAD catalog ID of the satellite")],
    lat: Annotated[float, typer.Argument(help="Latitude in degrees (-90 to 90)")],
    lon: Annotated[float, typer.Argument(help="Longitude in degrees (-180 to 180)")],
    alt: Annotated[
        float, typer.Option(help="Altitude above WGS84 ellipsoid in meters")
    ] = 0.0,
    start_time: Annotated[
        Optional[str],
        typer.Option(
            help="Start time (ISO-8601 or epoch string). Default: now if end_time not given"
        ),
    ] = None,
    end_time: Annotated[
        Optional[str], typer.Option(help="End time (ISO-8601 or epoch string)")
    ] = None,
    duration: Annotated[
        Optional[float], typer.Option(help="Duration in days (default: 7)")
    ] = None,
    min_elevation: Annotated[
        float, typer.Option(help="Minimum elevation angle in degrees")
    ] = 10.0,
    max_results: Annotated[
        Optional[int], typer.Option(help="Maximum number of access windows to display")
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option(help="Output format: 'rich' or 'simple'")
    ] = OutputFormat.rich,
    output_file: Annotated[
        Optional[Path], typer.Option(help="Path to export results as JSON")
    ] = None,
):
    """
    Compute satellite access windows for a ground location.

    Examples:

        # Compute next 7 days of ISS passes over New York City
        brahe access compute 25544 40.7128 -74.0060

        # Custom time range and elevation for GPS satellite
        brahe access compute 32260 40.7128 -74.0060 --start-time "2024-01-01T00:00:00" --duration 1 --min-elevation 15

        # Simple output and export to JSON
        brahe access compute 25544 40.7128 -74.0060 --output-format simple --output-file passes.json
    """
    # Initialize EOP
    set_cli_eop()

    # Validate latitude and longitude
    if not (-90 <= lat <= 90):
        typer.echo(f"Error: Latitude must be between -90 and 90 degrees (got {lat})")
        raise typer.Exit(code=1)
    if not (-180 <= lon <= 180):
        typer.echo(f"Error: Longitude must be between -180 and 180 degrees (got {lon})")
        raise typer.Exit(code=1)

    # Determine time range
    if end_time and not start_time:
        typer.echo("Error: Cannot specify --end-time without --start-time")
        raise typer.Exit(code=1)

    try:
        if start_time:
            epoch_start = brahe.Epoch(start_time)
        else:
            epoch_start = brahe.Epoch.now()

        if end_time:
            epoch_end = brahe.Epoch(end_time)
        else:
            duration_days = duration if duration is not None else 7.0
            epoch_end = epoch_start + (duration_days * 86400.0)

        if epoch_end <= epoch_start:
            typer.echo("Error: End time must be after start time")
            raise typer.Exit(code=1)

    except Exception as e:
        typer.echo(f"Error parsing time: {e}")
        raise typer.Exit(code=1)

    # Fetch TLE from CelesTrak
    try:
        sat_name, line1, line2 = brahe.datasets.celestrak.get_tle_by_id(norad_id)
    except Exception as e:
        typer.echo(f"Error fetching TLE for NORAD ID {norad_id}: {e}")
        raise typer.Exit(code=1)

    # Create propagator from TLE
    try:
        tle = brahe.TLE(sat_name, line1, line2)
        propagator = brahe.SGPPropagator(tle, 60.0)
    except Exception as e:
        typer.echo(f"Error creating propagator: {e}")
        raise typer.Exit(code=1)

    # Create location (convert degrees to radians)
    location = brahe.PointLocation(math.radians(lon), math.radians(lat), alt).with_name(
        f"Location ({lat:.4f}°, {lon:.4f}°)"
    )

    # Create constraint
    constraint = brahe.ElevationConstraint(min_elevation_deg=min_elevation)

    # Compute access windows
    try:
        windows = brahe.location_accesses(
            location, propagator, epoch_start, epoch_end, constraint
        )
    except Exception as e:
        typer.echo(f"Error computing access windows: {e}")
        raise typer.Exit(code=1)

    # Limit results if requested
    if max_results is not None and len(windows) > max_results:
        windows = windows[:max_results]

    # Display results
    if len(windows) == 0:
        typer.echo(f"\nNo access windows found for {sat_name} (NORAD ID: {norad_id})")
        typer.echo(f"  Location: {lat:.4f}° lat, {lon:.4f}° lon, {alt:.0f} m alt")
        typer.echo(f"  Time range: {epoch_start} to {epoch_end}")
        typer.echo(f"  Minimum elevation: {min_elevation:.1f}°")
        return

    if output_format == OutputFormat.rich:
        _display_rich(windows, sat_name, norad_id, lat, lon, alt, min_elevation)
    else:
        _display_simple(windows, sat_name, norad_id, lat, lon, alt, min_elevation)

    # Export to JSON if requested
    if output_file:
        try:
            _export_json(
                windows, output_file, sat_name, norad_id, lat, lon, alt, min_elevation
            )
            typer.echo(f"\n✓ Exported {len(windows)} access windows to {output_file}")
        except Exception as e:
            typer.echo(f"\nError exporting to JSON: {e}", err=True)
            raise typer.Exit(code=1)


def _display_rich(
    windows: list,
    sat_name: str,
    norad_id: int,
    lat: float,
    lon: float,
    alt: float,
    min_elevation: float,
):
    """Display access windows using Rich formatting."""
    console = Console()

    # Header
    console.print(
        f"\n[bold]Access Windows for {sat_name} (NORAD ID: {norad_id})[/bold]"
    )
    console.print(f"Location: {lat:.4f}° lat, {lon:.4f}° lon, {alt:.0f} m alt")
    console.print(f"Minimum elevation: {min_elevation:.1f}°")
    console.print(f"Found {len(windows)} access window(s)\n")

    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Start Time (UTC)", style="green")
    table.add_column("End Time (UTC)", style="red")
    table.add_column("Duration", style="yellow")
    table.add_column("Max Elev", justify="right")
    table.add_column("Az Open", justify="right")
    table.add_column("Az Close", justify="right")

    for window in windows:
        duration = window.window_close - window.window_open
        table.add_row(
            str(window.window_open),
            str(window.window_close),
            brahe.format_time_string(duration),
            f"{window.properties.elevation_max:.1f}°",
            f"{window.properties.azimuth_open:.0f}°",
            f"{window.properties.azimuth_close:.0f}°",
        )

    console.print(table)


def _display_simple(
    windows: list,
    sat_name: str,
    norad_id: int,
    lat: float,
    lon: float,
    alt: float,
    min_elevation: float,
):
    """Display access windows in simple text format."""
    typer.echo(f"\nAccess Windows for {sat_name} (NORAD ID: {norad_id})")
    typer.echo(f"Location: {lat:.4f}° lat, {lon:.4f}° lon, {alt:.0f} m alt")
    typer.echo(f"Minimum elevation: {min_elevation:.1f}°")
    typer.echo(f"Found {len(windows)} access window(s)\n")

    for i, window in enumerate(windows, 1):
        duration = window.window_close - window.window_open
        typer.echo(
            f"{i}. {window.window_open} | {window.window_close} | "
            f"{brahe.format_time_string(duration, short=True)} | "
            f"Max Elev: {window.properties.elevation_max:.1f}° | "
            f"Az: {window.properties.azimuth_open:.0f}°-{window.properties.azimuth_close:.0f}°"
        )


def _export_json(
    windows: list,
    output_file: Path,
    sat_name: str,
    norad_id: int,
    lat: float,
    lon: float,
    alt: float,
    min_elevation: float,
):
    """Export access windows to JSON file."""
    data = {
        "satellite": {"name": sat_name, "norad_id": norad_id},
        "location": {"latitude_deg": lat, "longitude_deg": lon, "altitude_m": alt},
        "constraint": {"min_elevation_deg": min_elevation},
        "windows": [
            {
                "window_open": str(window.window_open),
                "window_close": str(window.window_close),
                "duration_sec": window.window_close - window.window_open,
                "properties": {
                    "azimuth_open_deg": window.properties.azimuth_open,
                    "azimuth_close_deg": window.properties.azimuth_close,
                    "elevation_min_deg": window.properties.elevation_min,
                    "elevation_max_deg": window.properties.elevation_max,
                    "off_nadir_min_deg": window.properties.off_nadir_min,
                    "off_nadir_max_deg": window.properties.off_nadir_max,
                    "local_time_sec": window.properties.local_time,
                    "look_direction": str(window.properties.look_direction),
                    "asc_dsc": str(window.properties.asc_dsc),
                },
            }
            for window in windows
        ],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
