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


app.add_typer(celestrak_app, name="celestrak")
