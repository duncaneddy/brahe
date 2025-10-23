import json
import math
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
import typer
from typing_extensions import Annotated
from loguru import logger
from rich.console import Console
from rich.table import Table

import brahe
from brahe.cli.utils import set_cli_eop

app = typer.Typer()


class OutputFormat(str, Enum):
    """Output format for access window display."""

    table = "table"
    rich = "rich"
    simple = "simple"


class SortBy(str, Enum):
    """Sort field for access windows."""

    contact_number = "contact_number"
    start_time = "start_time"
    end_time = "end_time"
    duration = "duration"
    max_elevation = "max_elevation"
    start_azimuth = "start_azimuth"
    end_azimuth = "end_azimuth"


class SortOrder(str, Enum):
    """Sort order for access windows."""

    ascending = "ascending"
    descending = "descending"


@app.command()
def compute(
    norad_id: Annotated[int, typer.Argument(help="NORAD catalog ID of the satellite")],
    lat: Annotated[
        Optional[float],
        typer.Option(help="Latitude in degrees (-90 to 90). Use with --lon."),
    ] = None,
    lon: Annotated[
        Optional[float],
        typer.Option(help="Longitude in degrees (-180 to 180). Use with --lat."),
    ] = None,
    alt: Annotated[
        float, typer.Option(help="Altitude above WGS84 ellipsoid in meters")
    ] = 0.0,
    gs_provider: Annotated[
        Optional[str],
        typer.Option(
            help="Groundstation provider (e.g., 'ksat', 'atlas', 'aws'). Use with --gs-name."
        ),
    ] = None,
    gs_name: Annotated[
        Optional[str],
        typer.Option(help="Groundstation name to lookup. Use with --gs-provider."),
    ] = None,
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
        OutputFormat, typer.Option(help="Output format: 'table', 'rich', or 'simple'")
    ] = OutputFormat.table,
    sort_by: Annotated[
        SortBy, typer.Option(help="Sort windows by field")
    ] = SortBy.start_time,
    sort_order: Annotated[
        SortOrder, typer.Option(help="Sort order: 'ascending' or 'descending'")
    ] = SortOrder.ascending,
    output_file: Annotated[
        Optional[Path], typer.Option(help="Path to export results as JSON")
    ] = None,
):
    """
    Compute satellite access windows for a ground location.

    Examples:

        # Compute next 7 days of ISS passes over New York City
        brahe access compute 25544 --lat 40.7128 --lon -74.0060

        # Custom time range and elevation for GPS satellite
        brahe access compute 32260 --lat 40.7128 --lon -74.0060 --start-time "2024-01-01T00:00:00" --duration 1 --min-elevation 15

        # Simple output and export to JSON
        brahe access compute 25544 --lat 40.7128 --lon -74.0060 --output-format simple --output-file passes.json

        # Use groundstation lookup
        brahe access compute 25544 --gs-provider ksat --gs-name "Svalbard"
    """
    logger.info(f"Computing access windows for NORAD ID {norad_id}")
    start_time_compute = time.time()

    # Initialize EOP
    logger.debug("Initializing EOP provider")
    set_cli_eop()

    # Validate input: either lat/lon or gs_provider/gs_name
    has_coords = lat is not None and lon is not None
    has_gs_lookup = gs_provider is not None and gs_name is not None

    if not has_coords and not has_gs_lookup:
        typer.echo(
            "Error: Must provide either lat/lon coordinates OR --gs-provider and --gs-name"
        )
        raise typer.Exit(code=1)

    if has_coords and has_gs_lookup:
        typer.echo(
            "Error: Cannot specify both lat/lon coordinates AND --gs-provider/--gs-name. Choose one."
        )
        raise typer.Exit(code=1)

    if (gs_provider is None) != (gs_name is None):
        typer.echo("Error: Must specify both --gs-provider and --gs-name together")
        raise typer.Exit(code=1)

    # Handle groundstation lookup
    location_name = None
    if has_gs_lookup:
        logger.debug(
            f"Looking up groundstation '{gs_name}' from provider '{gs_provider}'"
        )
        try:
            stations = brahe.datasets.groundstations.load(gs_provider)
        except Exception as e:
            typer.echo(
                f"Error loading groundstations from provider '{gs_provider}': {e}"
            )
            raise typer.Exit(code=1)

        # Find station by name (case-insensitive)
        gs_name_upper = gs_name.upper()
        matching_stations = [
            s
            for s in stations
            if s.get_name() and gs_name_upper in s.get_name().upper()
        ]

        if not matching_stations:
            typer.echo(
                f"Error: No groundstation matching '{gs_name}' found in provider '{gs_provider}'"
            )
            raise typer.Exit(code=1)

        if len(matching_stations) > 1:
            typer.echo(f"Error: Multiple groundstations match '{gs_name}':")
            for station in matching_stations:
                typer.echo(f"  - {station.get_name()}")
            typer.echo("\nPlease provide a more specific name.")
            raise typer.Exit(code=1)

        station = matching_stations[0]
        # Extract coordinates (already in degrees)
        lat = station.lat
        lon = station.lon
        alt = station.alt
        location_name = station.get_name()
        logger.info(
            f"Found groundstation: {location_name} ({lat:.4f}°, {lon:.4f}°, {alt:.0f}m)"
        )

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
            # Get current time
            dt = datetime.now(timezone.utc)
            epoch_start = brahe.Epoch.from_datetime(
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                dt.second,
                0.0,
                brahe.TimeSystem.UTC,
            )

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
    logger.debug(f"Fetching TLE for NORAD ID {norad_id} from CelesTrak")
    try:
        sat_name, line1, line2 = brahe.datasets.celestrak.get_tle_by_id(norad_id)
        logger.info(f"Retrieved TLE for {sat_name}")
    except Exception as e:
        typer.echo(f"Error fetching TLE for NORAD ID {norad_id}: {e}")
        raise typer.Exit(code=1)

    # Create propagator from TLE
    try:
        propagator = brahe.SGPPropagator.from_3le(sat_name, line1, line2, 60.0)
    except Exception as e:
        typer.echo(f"Error creating propagator: {e}")
        raise typer.Exit(code=1)

    # Create location (convert degrees to radians)
    if location_name:
        location = brahe.PointLocation(
            math.radians(lon), math.radians(lat), alt
        ).with_name(location_name)
    else:
        location = brahe.PointLocation(
            math.radians(lon), math.radians(lat), alt
        ).with_name(f"Location ({lat:.4f}°, {lon:.4f}°)")

    # Create constraint
    constraint = brahe.ElevationConstraint(min_elevation_deg=min_elevation)

    # Compute access windows
    duration_days = (epoch_end - epoch_start) / 86400.0
    logger.info(
        f"Computing access windows from {epoch_start} to {epoch_end} ({duration_days:.1f} days)"
    )
    logger.debug(f"Constraint: min_elevation={min_elevation}°")
    try:
        windows = brahe.location_accesses(
            location, propagator, epoch_start, epoch_end, constraint
        )
        elapsed = time.time() - start_time_compute
        logger.info(f"Found {len(windows)} access windows in {elapsed:.2f}s")
    except Exception as e:
        typer.echo(f"Error computing access windows: {e}")
        raise typer.Exit(code=1)

    # Sort windows based on user preference
    reverse = sort_order == SortOrder.descending
    if sort_by == SortBy.start_time:
        windows.sort(key=lambda w: w.window_open, reverse=reverse)
    elif sort_by == SortBy.end_time:
        windows.sort(key=lambda w: w.window_close, reverse=reverse)
    elif sort_by == SortBy.duration:
        windows.sort(key=lambda w: w.window_close - w.window_open, reverse=reverse)
    elif sort_by == SortBy.max_elevation:
        windows.sort(key=lambda w: w.properties.elevation_max, reverse=reverse)
    elif sort_by == SortBy.start_azimuth:
        windows.sort(key=lambda w: w.properties.azimuth_open, reverse=reverse)
    elif sort_by == SortBy.end_azimuth:
        windows.sort(key=lambda w: w.properties.azimuth_close, reverse=reverse)
    # contact_number doesn't require sorting as it's the original order

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

    if output_format == OutputFormat.table:
        _display_table(windows, sat_name, norad_id, lat, lon, alt, min_elevation)
    elif output_format == OutputFormat.rich:
        _display_rich(windows, sat_name, norad_id, lat, lon, alt, min_elevation)
    else:
        _display_simple(windows, sat_name, norad_id, lat, lon, alt, min_elevation)

    # Export to JSON if requested
    if output_file:
        logger.debug(f"Exporting results to {output_file}")
        try:
            _export_json(
                windows, output_file, sat_name, norad_id, lat, lon, alt, min_elevation
            )
            logger.info(f"Exported {len(windows)} windows to {output_file}")
            typer.echo(f"\n✓ Exported {len(windows)} access windows to {output_file}")
        except Exception as e:
            typer.echo(f"\nError exporting to JSON: {e}", err=True)
            raise typer.Exit(code=1)


def _display_table(
    windows: list,
    sat_name: str,
    norad_id: int,
    lat: float,
    lon: float,
    alt: float,
    min_elevation: float,
):
    """Display access windows in table format with contact numbers."""
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
    table.add_column("Contact #", justify="right", style="cyan")
    table.add_column("Start Time (UTC)", style="green")
    table.add_column("End Time (UTC)", style="red")
    table.add_column("Duration", style="yellow")
    table.add_column("Max Elev (deg)", justify="right")
    table.add_column("Start Az (deg)", justify="right")
    table.add_column("End Az (deg)", justify="right")

    for i, window in enumerate(windows, 1):
        duration = window.window_close - window.window_open
        table.add_row(
            str(i),
            str(window.window_open),
            str(window.window_close),
            brahe.format_time_string(duration),
            f"{window.properties.elevation_max:.1f}",
            f"{window.properties.azimuth_open:.0f}",
            f"{window.properties.azimuth_close:.0f}",
        )

    console.print(table)


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
    table.add_column("Max Elev (deg)", justify="right")
    table.add_column("Az Open (deg)", justify="right")
    table.add_column("Az Close (deg)", justify="right")

    for window in windows:
        duration = window.window_close - window.window_open
        table.add_row(
            str(window.window_open),
            str(window.window_close),
            brahe.format_time_string(duration),
            f"{window.properties.elevation_max:.1f}",
            f"{window.properties.azimuth_open:.0f}",
            f"{window.properties.azimuth_close:.0f}",
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
                "contact_number": i,
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
            for i, window in enumerate(windows, 1)
        ],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
