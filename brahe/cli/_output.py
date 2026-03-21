"""
Shared CLI output formatting for GP records, SATCAT records, and other query results.

Provides column definitions, derived value computation, and multi-format rendering
(Rich table, markdown table, JSON, OMM) used by both celestrak and spacetrack CLI modules.
"""

import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

import brahe as bh


class CLIOutputFormat(str, Enum):
    """Output format for query results."""

    rich = "rich"
    markdown = "markdown"
    json = "json"
    omm = "omm"


# ---------------------------------------------------------------------------
# GP Record column definitions
# Format: (header, width, formatter_function)
# ---------------------------------------------------------------------------

GP_COLUMNS = {
    "name": ("Name", 20, str),
    "norad_id": ("ID", 6, str),
    "epoch": ("Epoch", 24, str),
    "age": ("Age", 10, lambda x: bh.format_time_string(x, short=True)),
    "sma": ("SMA (km)", 9, lambda x: f"{x / 1000.0:.1f}"),
    "perigee": ("Peri (km)", 10, lambda x: f"{x / 1000.0:.1f}"),
    "apogee": ("Apo (km)", 9, lambda x: f"{x / 1000.0:.1f}"),
    "ecc": ("Ecc", 10, lambda x: f"{x:.6f}"),
    "inc": ("Inc (deg)", 9, lambda x: f"{x:.2f}"),
    "raan": ("RAAN (deg)", 10, lambda x: f"{x:.2f}"),
    "argp": ("ArgP (deg)", 10, lambda x: f"{x:.2f}"),
    "ma": ("MA (deg)", 9, lambda x: f"{x:.2f}"),
    "period": ("Period (min)", 12, lambda x: f"{x:.1f}"),
    "mean_motion": ("n (rev/day)", 11, lambda x: f"{x:.4f}"),
    "country": ("Country", 8, str),
    "object_type": ("Type", 12, str),
}

GP_COLUMN_PRESETS = {
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
    "all": list(GP_COLUMNS.keys()),
}


# ---------------------------------------------------------------------------
# SATCAT column definitions (SpaceTrack SATCATRecord fields)
# ---------------------------------------------------------------------------

SATCAT_COLUMNS = {
    "satname": ("Name", 24, str),
    "norad_cat_id": ("ID", 6, lambda x: str(x) if x is not None else "?"),
    "intldes": ("IntlDes", 12, str),
    "object_type": ("Type", 14, str),
    "country": ("Country", 8, str),
    "launch": ("Launch", 12, str),
    "decay": ("Decay", 12, str),
    "period": ("Period (min)", 12, str),
    "inclination": ("Inc (deg)", 10, str),
    "apogee": ("Apo (km)", 9, str),
    "perigee": ("Peri (km)", 9, str),
    "rcs_size": ("RCS Size", 9, str),
}

SATCAT_COLUMN_PRESETS = {
    "minimal": ["satname", "norad_cat_id"],
    "default": [
        "satname",
        "norad_cat_id",
        "object_type",
        "country",
        "launch",
        "period",
        "inclination",
        "apogee",
        "perigee",
    ],
    "all": list(SATCAT_COLUMNS.keys()),
}


# ---------------------------------------------------------------------------
# CelesTrak SATCAT column definitions (CelestrakSATCATRecord fields)
# ---------------------------------------------------------------------------

CELESTRAK_SATCAT_COLUMNS = {
    "satname": ("Name", 24, str),
    "norad_cat_id": ("ID", 6, lambda x: str(x) if x is not None else "?"),
    "intldes": ("IntlDes", 12, str),
    "object_type": ("Type", 14, str),
    "country": ("Country", 8, str),
    "launch": ("Launch", 12, str),
    "decay": ("Decay", 12, str),
    "period": ("Period (min)", 12, str),
    "inclination": ("Inc (deg)", 10, str),
    "apogee": ("Apo (km)", 9, str),
    "perigee": ("Peri (km)", 9, str),
    "rcs_size": ("RCS Size", 9, str),
}


def parse_columns(
    columns: Optional[str],
    available: dict,
    presets: dict,
) -> List[str]:
    """Parse column specification into a validated list of column keys.

    Args:
        columns: Preset name, comma-separated column list, or None for default.
        available: Dict of available column definitions.
        presets: Dict of column preset names to column key lists.

    Returns:
        List of validated column keys.

    Raises:
        typer.Exit: If invalid column names are specified.
    """
    if columns is None:
        return presets["default"]
    if columns.lower() in presets:
        return presets[columns.lower()]

    column_list = [col.strip() for col in columns.split(",")]
    invalid = [col for col in column_list if col not in available]
    if invalid:
        console = Console()
        console.print(f"[red]ERROR: Invalid column(s): {', '.join(invalid)}[/red]")
        console.print(
            f"[yellow]Available columns: {', '.join(available.keys())}[/yellow]"
        )
        raise typer.Exit(code=1)
    return column_list


def _current_epoch():
    """Get current time as a brahe Epoch."""
    now = datetime.now()
    return bh.Epoch.from_datetime(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second + now.microsecond / 1e6,
        0,
        bh.TimeSystem.UTC,
    )


def compute_gp_row(record, current_epoch) -> dict:
    """Compute a row dict with raw and derived values from a GPRecord.

    Derives SMA from mean motion, computes period, perigee/apogee altitude,
    and age since epoch.

    Args:
        record: A brahe GPRecord object.
        current_epoch: Current time as a brahe Epoch for age computation.

    Returns:
        Dict with keys matching GP_COLUMNS.
    """
    name = record.object_name or "UNKNOWN"
    norad_id = record.norad_cat_id if record.norad_cat_id is not None else "?"
    mean_motion_val = record.mean_motion
    ecc = record.eccentricity

    # Derive SMA from mean motion (rev/day -> rad/s -> SMA via Kepler)
    if mean_motion_val is not None and mean_motion_val > 0:
        n_rad_s = mean_motion_val * 2.0 * math.pi / 86400.0
        a = (bh.GM_EARTH / (n_rad_s**2)) ** (1.0 / 3.0)
    else:
        a = 0.0

    period_sec = bh.orbital_period(a) if a > 0 else 0.0
    period_min = period_sec / 60.0
    perigee_alt_m = (a * (1.0 - (ecc or 0.0)) - bh.R_EARTH) if a > 0 else 0.0
    apogee_alt_m = (a * (1.0 + (ecc or 0.0)) - bh.R_EARTH) if a > 0 else 0.0

    epoch_str = record.epoch or ""
    age_seconds = 0.0
    if epoch_str:
        try:
            ep = bh.Epoch.from_string(epoch_str)
            age_seconds = current_epoch - ep
        except Exception:
            pass

    return {
        "name": name,
        "norad_id": norad_id,
        "epoch": epoch_str,
        "age": age_seconds,
        "sma": a,
        "perigee": perigee_alt_m,
        "apogee": apogee_alt_m,
        "ecc": ecc or 0.0,
        "inc": record.inclination or 0.0,
        "raan": record.ra_of_asc_node or 0.0,
        "argp": record.arg_of_pericenter or 0.0,
        "ma": record.mean_anomaly or 0.0,
        "period": period_min,
        "mean_motion": mean_motion_val or 0.0,
        "country": record.country_code or "",
        "object_type": record.object_type or "",
    }


def _build_rich_table(
    rows: List[dict], column_list: List[str], columns_def: dict
) -> Table:
    """Build a Rich Table from row dicts and column definitions."""
    table = Table(show_header=True, header_style="bold magenta")
    for col_key in column_list:
        header, width, _ = columns_def[col_key]
        table.add_column(header, style="cyan", width=width)

    for row in rows:
        values = []
        for col_key in column_list:
            _, _, formatter = columns_def[col_key]
            values.append(formatter(row[col_key]))
        table.add_row(*values)

    return table


def _build_markdown_table(
    rows: List[dict], column_list: List[str], columns_def: dict
) -> str:
    """Build a markdown table string from row dicts and column definitions."""
    headers = [columns_def[col][0] for col in column_list]
    widths = [columns_def[col][1] for col in column_list]

    # Header row
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    separator = "| " + " | ".join("-" * w for w in widths) + " |"

    # Data rows
    data_lines = []
    for row in rows:
        cells = []
        for col_key, w in zip(column_list, widths):
            _, _, formatter = columns_def[col_key]
            cells.append(str(formatter(row[col_key])).ljust(w))
        data_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_line, separator] + data_lines)


def format_gp_records(
    records: list,
    output_format: CLIOutputFormat,
    columns: Optional[str],
    output_file: Optional[Path],
    console: Console,
):
    """Format and display/save GP records in the specified format.

    Args:
        records: List of GPRecord objects.
        output_format: Output format (rich, markdown, json, omm).
        columns: Column preset or comma-separated list (for table formats).
        output_file: Optional path to write output to instead of stdout.
        console: Rich Console for output.
    """
    if not records:
        console.print("[yellow]No records found.[/yellow]")
        return

    if output_format == CLIOutputFormat.json:
        data = [record.to_dict() for record in records]
        output = json.dumps(data, indent=2, default=str)
        if output_file:
            output_file.write_text(output)
            console.print(f"Wrote {len(records)} record(s) to {output_file}")
        else:
            typer.echo(output)
        return

    if output_format == CLIOutputFormat.omm:
        omm_list = []
        for record in records:
            try:
                omm = record.to_omm()
                omm_json = omm.to_json_string(bh.CCSDSJsonKeyCase.Lower)
                omm_list.append(json.loads(omm_json))
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not convert record to OMM: {e}[/yellow]"
                )
        output = json.dumps(omm_list, indent=2)
        if output_file:
            output_file.write_text(output)
            console.print(f"Wrote {len(omm_list)} OMM record(s) to {output_file}")
        else:
            typer.echo(output)
        return

    # Table formats (rich or markdown)
    column_list = parse_columns(columns, GP_COLUMNS, GP_COLUMN_PRESETS)
    current_epoch = _current_epoch()

    rows = []
    for record in records:
        row = compute_gp_row(record, current_epoch)
        rows.append(row)

    if output_format == CLIOutputFormat.markdown:
        md = _build_markdown_table(rows, column_list, GP_COLUMNS)
        if output_file:
            output_file.write_text(md)
            console.print(f"Wrote {len(records)} record(s) to {output_file}")
        else:
            typer.echo(md)
    else:
        # Rich table (default)
        table = _build_rich_table(rows, column_list, GP_COLUMNS)
        if output_file:
            # Write plain text table to file
            with open(output_file, "w") as f:
                file_console = Console(file=f, width=200)
                file_console.print(table)
            console.print(f"Wrote {len(records)} record(s) to {output_file}")
        else:
            console.print()
            console.print(table)

    console.print(f"\n[dim]{len(records)} record(s)[/dim]")


def format_satcat_records(
    records: list,
    output_format: CLIOutputFormat,
    columns: Optional[str],
    output_file: Optional[Path],
    console: Console,
    columns_def: dict = None,
    presets: dict = None,
):
    """Format and display/save SATCAT records in the specified format.

    Works with both SpaceTrack SATCATRecord and CelesTrak CelestrakSATCATRecord.

    Args:
        records: List of SATCAT record objects.
        output_format: Output format.
        columns: Column preset or comma-separated list.
        output_file: Optional path to write output to.
        console: Rich Console for output.
        columns_def: Column definitions dict (defaults to SATCAT_COLUMNS).
        presets: Column presets dict (defaults to SATCAT_COLUMN_PRESETS).
    """
    if columns_def is None:
        columns_def = SATCAT_COLUMNS
    if presets is None:
        presets = SATCAT_COLUMN_PRESETS

    if not records:
        console.print("[yellow]No records found.[/yellow]")
        return

    if output_format in (CLIOutputFormat.json, CLIOutputFormat.omm):
        # SATCAT records don't support OMM; treat omm as json for SATCAT
        data = []
        for rec in records:
            row = {}
            for key in columns_def:
                row[key] = getattr(rec, key, None)
            data.append(row)
        output = json.dumps(data, indent=2, default=str)
        if output_file:
            output_file.write_text(output)
            console.print(f"Wrote {len(records)} record(s) to {output_file}")
        else:
            typer.echo(output)
        return

    # Table formats
    column_list = parse_columns(columns, columns_def, presets)

    rows = []
    for rec in records:
        row = {}
        for key in columns_def:
            val = getattr(rec, key, None)
            row[key] = val if val is not None else ""
        rows.append(row)

    if output_format == CLIOutputFormat.markdown:
        md = _build_markdown_table(rows, column_list, columns_def)
        if output_file:
            output_file.write_text(md)
            console.print(f"Wrote {len(records)} record(s) to {output_file}")
        else:
            typer.echo(md)
    else:
        table = _build_rich_table(rows, column_list, columns_def)
        if output_file:
            with open(output_file, "w") as f:
                file_console = Console(file=f, width=200)
                file_console.print(table)
            console.print(f"Wrote {len(records)} record(s) to {output_file}")
        else:
            console.print()
            console.print(table)

    console.print(f"\n[dim]{len(records)} record(s)[/dim]")


def parse_filters(filter_args: Optional[List[str]]) -> List[tuple]:
    """Parse --filter arguments into (field, value) tuples.

    Each filter string should be "FIELD VALUE", split on first whitespace.
    Example: "INCLINATION >50" -> ("INCLINATION", ">50")

    Args:
        filter_args: List of filter strings from CLI.

    Returns:
        List of (field, value) tuples.
    """
    if not filter_args:
        return []
    result = []
    for f in filter_args:
        parts = f.split(maxsplit=1)
        if len(parts) != 2:
            console = Console()
            console.print(
                f'[red]ERROR: Invalid filter format: "{f}". '
                f'Expected "FIELD VALUE" (e.g., "INCLINATION >50")[/red]'
            )
            raise typer.Exit(code=1)
        result.append((parts[0], parts[1]))
    return result
