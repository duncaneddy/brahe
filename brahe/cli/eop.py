from enum import Enum
import typer
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn
import brahe



class ProductType(str, Enum):
    standard = "standard"
    c04 = "c04"

app = typer.Typer()

@app.command()
def download(filepath: Annotated[str, typer.Argument(..., help="Filepath to output data")],
             type: Annotated[ProductType, typer.Option(..., help="Type of data product to download")]):
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
    ) as progress:
        progress.add_task(description="Downloading...", total=None)
        if type == ProductType.standard:
            brahe.download_standard_eop_file(filepath)
        else:
            brahe.download_c04_eop_file(filepath)
    typer.echo(f"Downloaded {type} EOP data to {filepath}")

@app.command()
def get_utc_ut1():
    typer.echo("Not Yet Implemented")

@app.command()
def get_polar_motion():
    typer.echo("Not Yet Implemented")

@app.command()
def get_cip_offset():
    typer.echo("Not Yet Implemented")

@app.command()
def get_lod():
    typer.echo("Not Yet Implemented")