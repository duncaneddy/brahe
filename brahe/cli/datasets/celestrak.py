"""
CLI commands for CelesTrak datasets
"""

from enum import Enum
from pathlib import Path
import typer
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import brahe.datasets as datasets
import brahe as bh


class ContentFormat(str, Enum):
    tle = "tle"
    three_le = "3le"


class FileFormat(str, Enum):
    txt = "txt"
    csv = "csv"
    json = "json"


app = typer.Typer()

# Column definitions for search command table output
# Format: (header, width, formatter_function)
AVAILABLE_COLUMNS = {
    "name": ("Name", 18, str),
    "norad_id": ("ID", 6, str),
    "epoch": ("Epoch", 20, str),
    "age": ("Age", 10, lambda x: bh.format_time_string(x, short=True)),
    "sma": ("SMA (km)", 9, lambda x: f"{x / 1000.0:.1f}"),
    "perigee": ("Peri (km)", 10, lambda x: f"{x / 1000.0:.1f}"),
    "apogee": ("Apo (km)", 9, lambda x: f"{x / 1000.0:.1f}"),
    "ecc": ("Ecc", 8, lambda x: f"{x:.6f}"),
    "inc": ("Inc (°)", 7, lambda x: f"{x:.2f}"),
    "raan": ("RAAN (°)", 9, lambda x: f"{x:.2f}"),
    "argp": ("ArgP (°)", 9, lambda x: f"{x:.2f}"),
    "ma": ("MA (°)", 7, lambda x: f"{x:.2f}"),
    "period": ("Period (min)", 12, lambda x: f"{x:.1f}"),
    "mean_motion": ("n", 6, lambda x: f"{x:.3f}"),
}

# Column presets for search command
COLUMN_PRESETS = {
    "minimal": ["name", "norad_id"],
    "default": [
        "name",
        "norad_id",
        "epoch",
        "age",
        "period",
        "sma",
        "ecc",
        "inc",
        "raan",
        "argp",
        "ma",
    ],
    "all": list(AVAILABLE_COLUMNS.keys()),
}


@app.command()
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


@app.command()
def lookup(
    name: Annotated[str, typer.Argument(help="Satellite name to search for")],
    group: Annotated[
        str | None, typer.Option(help="Optional satellite group to search first")
    ] = None,
):
    """
    Lookup a satellite by name and display its NORAD ID and TLE.

    Uses cascading search: specified group → "active" group → CelesTrak NAME API.

    Examples:
        brahe datasets celestrak lookup "ISS"
        brahe datasets celestrak lookup "STARLINK-1234" --group active
        brahe datasets celestrak lookup "GPS" --group gnss
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Searching for '{name}'...", total=None)

        try:
            sat_name, line1, line2 = datasets.celestrak.get_tle_by_name(name, group)
        except Exception as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(code=1)

    # Extract NORAD ID from TLE line 1 (columns 2-7)
    norad_id = line1[2:7].strip()

    typer.echo(f"\n{sat_name} [NORAD ID: {norad_id}]")
    typer.echo("\nTLE Lines:")
    typer.echo(f"  {line1}")
    typer.echo(f"  {line2}")
    typer.echo(f"\n✓ Found satellite '{sat_name}'")


@app.command()
def show(
    identifier: Annotated[
        str, typer.Argument(help="Satellite identifier (NORAD ID or name)")
    ],
    group: Annotated[
        str | None, typer.Option(help="Optional satellite group to search")
    ] = None,
    compact: Annotated[
        bool, typer.Option("--compact", "-c", help="Show only TLE lines")
    ] = False,
    simple: Annotated[
        bool,
        typer.Option(
            "--simple", "-s", help="Use simple text formatting instead of rich"
        ),
    ] = False,
):
    """
    Display TLE information and computed orbital parameters for a satellite.

    The identifier can be either a NORAD ID (numeric) or satellite name.

    Examples:
        brahe datasets celestrak show 25544
        brahe datasets celestrak show "ISS" --group stations
        brahe datasets celestrak show 25544 --compact
        brahe datasets celestrak show 25544 --simple
    """
    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Fetching TLE for '{identifier}'...", total=None)

        try:
            # Try to parse as NORAD ID first
            try:
                norad_id = int(identifier)
                sat_name, line1, line2 = datasets.celestrak.get_tle_by_id(
                    norad_id, group
                )
            except ValueError:
                # Not a number, treat as name
                sat_name, line1, line2 = datasets.celestrak.get_tle_by_name(
                    identifier, group
                )
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]", style="bold")
            raise typer.Exit(code=1)

    # If compact mode, just show TLE lines
    if compact:
        console.print(line1)
        console.print(line2)
        return

    # Parse TLE to get orbital elements
    try:
        epoch, elements = bh.keplerian_elements_from_tle(line1, line2)
    except Exception as e:
        console.print(f"[red]ERROR: Failed to parse TLE: {e}[/red]", style="bold")
        raise typer.Exit(code=1)

    # Extract NORAD ID
    norad_id = line1[2:7].strip()

    # Extract orbital elements (elements are in [a, e, i, Ω, ω, M])
    # Note: angles in TLE are in degrees
    a = elements[0]  # meters
    e = elements[1]  # dimensionless
    inc = elements[2]  # degrees
    raan = elements[3]  # degrees
    argp = elements[4]  # degrees
    ma = elements[5]  # degrees

    # Compute derived parameters
    period_sec = bh.orbital_period(a)
    period_min = period_sec / 60.0
    period_hr = period_sec / 3600.0
    mean_motion = 86400.0 / period_sec  # rev/day

    # Compute altitudes (convert meters to km)
    perigee_alt_km = (a * (1.0 - e) - bh.R_EARTH) / 1000.0
    apogee_alt_km = (a * (1.0 + e) - bh.R_EARTH) / 1000.0

    # Compute ephemeris age
    from datetime import datetime

    now = datetime.now()
    current_epoch = bh.Epoch.from_datetime(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second + now.microsecond / 1e6,
        0,
        bh.TimeSystem.UTC,
    )
    age_seconds = current_epoch - epoch
    age_str = bh.format_time_string(age_seconds, short=True)

    # Display with simple or rich formatting
    if simple:
        # Simple text output
        typer.echo(f"\n{sat_name} [NORAD ID: {norad_id}]")
        typer.echo("\nTLE Lines:")
        typer.echo(f"  Line 1: {line1}")
        typer.echo(f"  Line 2: {line2}")

        typer.echo("\nOrbital Elements:")
        typer.echo(f"  Epoch:              {epoch}")
        typer.echo(f"  Ephemeris Age:      {age_str}")
        typer.echo(f"  Semi-major axis:    {a / 1000.0:.1f} km")
        typer.echo(f"  Eccentricity:       {e:.7f}")
        typer.echo(f"  Inclination:        {inc:.4f}°")
        typer.echo(f"  RAAN:               {raan:.4f}°")
        typer.echo(f"  Arg of Perigee:     {argp:.4f}°")
        typer.echo(f"  Mean Anomaly:       {ma:.4f}°")

        typer.echo("\nOrbital Characteristics:")
        typer.echo(
            f"  Orbital Period:     {period_min:.1f} min ({period_hr:.2f} hours)"
        )
        typer.echo(f"  Mean Motion:        {mean_motion:.3f} rev/day")
        typer.echo(f"  Perigee Altitude:   {perigee_alt_km:.1f} km")
        typer.echo(f"  Apogee Altitude:    {apogee_alt_km:.1f} km")
    else:
        # Rich formatting
        console.print()
        console.print(
            Panel(
                f"[bold cyan]{sat_name}[/bold cyan]\n[dim]NORAD ID: {norad_id}[/dim]",
                border_style="cyan",
            )
        )

        # TLE Lines table
        tle_table = Table(show_header=False, box=None, padding=(0, 1))
        tle_table.add_column("Label", style="dim")
        tle_table.add_column("Value", style="yellow")
        tle_table.add_row("Line 1:", line1)
        tle_table.add_row("Line 2:", line2)

        console.print(
            Panel(tle_table, title="[bold]TLE Lines[/bold]", border_style="blue")
        )

        # Orbital Elements table
        elements_table = Table(show_header=False, box=None, padding=(0, 1))
        elements_table.add_column("Parameter", style="cyan")
        elements_table.add_column("Value", justify="right")

        elements_table.add_row("Epoch", str(epoch))
        elements_table.add_row("Ephemeris Age", f"[yellow]{age_str}[/yellow]")
        elements_table.add_row("Semi-major axis", f"{a / 1000.0:.1f} km")
        elements_table.add_row("Eccentricity", f"{e:.7f}")
        elements_table.add_row("Inclination", f"{inc:.4f}°")
        elements_table.add_row("RAAN", f"{raan:.4f}°")
        elements_table.add_row("Arg of Perigee", f"{argp:.4f}°")
        elements_table.add_row("Mean Anomaly", f"{ma:.4f}°")

        console.print(
            Panel(
                elements_table,
                title="[bold]Orbital Elements[/bold]",
                border_style="green",
            )
        )

        # Orbital Characteristics table
        char_table = Table(show_header=False, box=None, padding=(0, 1))
        char_table.add_column("Parameter", style="cyan")
        char_table.add_column("Value", justify="right")

        char_table.add_row(
            "Orbital Period", f"{period_min:.1f} min ({period_hr:.2f} hours)"
        )
        char_table.add_row("Mean Motion", f"{mean_motion:.3f} rev/day")
        char_table.add_row(
            "Perigee Altitude", f"[green]{perigee_alt_km:.1f} km[/green]"
        )
        char_table.add_row("Apogee Altitude", f"[green]{apogee_alt_km:.1f} km[/green]")

        console.print(
            Panel(
                char_table,
                title="[bold]Orbital Characteristics[/bold]",
                border_style="magenta",
            )
        )
        console.print()


@app.command("list-groups")
def list_groups():
    """
    List commonly used CelesTrak satellite groups.

    For a complete list of available groups, visit:
    https://celestrak.org/NORAD/elements/

    Examples:
        brahe datasets celestrak list-groups
    """
    console = Console()

    # Define common groups with descriptions
    groups = [
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
        ("weather", "Weather satellites"),
        ("geodetic", "Geodetic satellites"),
        ("cosmos-2251-debris", "Debris from Cosmos 2251 collision"),
        ("iridium-33-debris", "Debris from Iridium 33 collision"),
        ("fengyun-1c-debris", "Debris from Fengyun-1C ASAT test"),
        ("cosmos-1408-debris", "Debris from Cosmos 1408 ASAT test"),
    ]

    console.print("\n[bold cyan]Common CelesTrak Satellite Groups[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Group Name", style="cyan", width=20)
    table.add_column("Description", style="white")

    for group_name, description in groups:
        table.add_row(group_name, description)

    console.print(table)
    console.print(
        f"\n[dim]Total: {len(groups)} groups shown. For complete list, visit https://celestrak.org/NORAD/elements/[/dim]\n"
    )


@app.command()
def search(
    pattern: Annotated[
        str, typer.Argument(help="Satellite name pattern to search for")
    ],
    group: Annotated[str, typer.Option(help="Satellite group to search")] = "active",
    table: Annotated[
        bool, typer.Option("--table", "-t", help="Display results as table")
    ] = False,
    columns: Annotated[
        str | None,
        typer.Option(
            help="Columns to display: 'minimal', 'default', 'all', or comma-separated list (e.g., 'name,norad_id,sma,inc')"
        ),
    ] = None,
):
    """
    Search for satellites by name pattern within a group.

    Default output is simple format (one satellite per line). Use --table for rich table output.

    Available columns for table output:
        name, norad_id, epoch, age, sma, perigee, apogee, ecc, inc, raan, argp, ma, period, mean_motion

    Column presets:
        minimal: name, norad_id
        default: name, norad_id, epoch, age, period, sma, ecc, inc, raan, argp, ma
        all: all available columns

    Examples:
        brahe datasets celestrak search "CAPELLA"
        brahe datasets celestrak search "STARLINK" --group starlink
        brahe datasets celestrak search "GPS" --group gnss --table
        brahe datasets celestrak search "ISS" --table --columns minimal
        brahe datasets celestrak search "STARLINK" --table --columns name,norad_id,inc,perigee,apogee
    """
    console = Console()

    # Download ephemeris for the group
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description=f"Searching '{pattern}' in {group} group...", total=None
        )

        try:
            ephemeris = datasets.celestrak.get_ephemeris(group)
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]", style="bold")
            raise typer.Exit(code=1)

    # Filter by pattern (case-insensitive substring match)
    pattern_upper = pattern.upper()
    matches = [
        (name, line1, line2)
        for name, line1, line2 in ephemeris
        if pattern_upper in name.upper()
    ]

    if not matches:
        console.print(
            f"[yellow]No satellites found matching '{pattern}' in group '{group}'[/yellow]"
        )
        raise typer.Exit(code=0)

    # Simple format (default)
    if not table:
        for name, line1, line2 in matches:
            norad_id = line1[2:7].strip()
            typer.echo(f"{name} (NORAD: {norad_id})")
        typer.echo(f"\n✓ Found {len(matches)} satellite(s)")
        return

    # Table format - determine columns to display
    if columns is None:
        column_list = COLUMN_PRESETS["default"]
    elif columns.lower() in COLUMN_PRESETS:
        column_list = COLUMN_PRESETS[columns.lower()]
    else:
        # Parse comma-separated column list
        column_list = [col.strip() for col in columns.split(",")]
        # Validate column names
        invalid_cols = [col for col in column_list if col not in AVAILABLE_COLUMNS]
        if invalid_cols:
            console.print(
                f"[red]ERROR: Invalid column(s): {', '.join(invalid_cols)}[/red]"
            )
            console.print(
                f"[yellow]Available columns: {', '.join(AVAILABLE_COLUMNS.keys())}[/yellow]"
            )
            raise typer.Exit(code=1)

    # Create rich table
    rich_table = Table(show_header=True, header_style="bold magenta")

    # Add columns to table
    for col_key in column_list:
        header, width, _ = AVAILABLE_COLUMNS[col_key]
        rich_table.add_column(header, style="cyan", width=width)

    # Parse each match and add to table
    from datetime import datetime

    now = datetime.now()
    current_epoch = bh.Epoch.from_datetime(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second + now.microsecond / 1e6,
        0,
        bh.TimeSystem.UTC,
    )

    for name, line1, line2 in matches:
        # Extract NORAD ID
        norad_id = line1[2:7].strip()

        # Parse TLE to get orbital elements
        try:
            epoch, elements = bh.keplerian_elements_from_tle(line1, line2)
        except Exception as e:
            # Skip satellites with invalid TLEs
            console.print(
                f"[dim]Warning: Skipping {name} - invalid TLE: {e}[/dim]", err=True
            )
            continue

        # Extract orbital elements
        a = elements[0]  # meters
        e = elements[1]  # dimensionless
        inc = elements[2]  # degrees
        raan = elements[3]  # degrees
        argp = elements[4]  # degrees
        ma = elements[5]  # degrees

        # Compute derived parameters
        period_sec = bh.orbital_period(a)
        period_min = period_sec / 60.0
        mean_motion = 86400.0 / period_sec  # rev/day
        perigee_alt_km = (a * (1.0 - e) - bh.R_EARTH) / 1000.0
        apogee_alt_km = (a * (1.0 + e) - bh.R_EARTH) / 1000.0
        age_seconds = current_epoch - epoch

        # Build row data dictionary
        row_data = {
            "name": name,
            "norad_id": norad_id,
            "epoch": str(epoch),
            "age": age_seconds,  # Will be formatted by column formatter
            "sma": a,  # meters, will be formatted by column formatter
            "perigee": perigee_alt_km * 1000.0,  # convert back to meters for formatter
            "apogee": apogee_alt_km * 1000.0,  # convert back to meters for formatter
            "ecc": e,
            "inc": inc,
            "raan": raan,
            "argp": argp,
            "ma": ma,
            "period": period_min,
            "mean_motion": mean_motion,
        }

        # Format and add row
        row_values = []
        for col_key in column_list:
            _, _, formatter = AVAILABLE_COLUMNS[col_key]
            value = row_data[col_key]
            formatted_value = formatter(value)
            row_values.append(formatted_value)

        rich_table.add_row(*row_values)

    # Display table
    console.print()
    console.print(rich_table)
    console.print(f"\n[dim]✓ Found {len(matches)} satellite(s)[/dim]\n")
