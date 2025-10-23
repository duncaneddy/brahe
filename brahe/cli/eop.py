from enum import Enum
from pathlib import Path
import typer
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn
import brahe


class ProductType(str, Enum):
    standard = "standard"
    c04 = "c04"


class ProductSource(str, Enum):
    default = "default"
    file = "file"


def get_global_eop_source(product: ProductType, source: ProductSource, filepath: Path):
    # Get EOP Data
    if source == ProductSource.default:
        if product == ProductType.standard:
            eop = brahe.FileEOPProvider.from_default_standard(True, "Error")
        else:
            eop = brahe.FileEOPProvider.from_default_c04(True, "Error")
    else:
        if product == ProductType.standard:
            eop = brahe.FileEOPProvider.from_standard_file(
                filepath.absolute().as_posix(), True, "Error"
            )
        else:
            eop = brahe.FileEOPProvider.from_c04_file(
                filepath.absolute().as_posix(), True, "Error"
            )

    return eop


app = typer.Typer()


@app.command()
def download(
    filepath: Annotated[Path, typer.Argument(..., help="Filepath to output data")],
    product: Annotated[
        ProductType, typer.Option(..., help="Type of data product to download")
    ],
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Downloading...", total=None)
        if product == ProductType.standard:
            brahe.download_standard_eop_file(filepath.absolute().as_posix())
        else:
            brahe.download_c04_eop_file(filepath.absolute().as_posix())
    typer.echo(f"Downloaded {product.value} EOP data to {filepath}")


@app.command()
def get_utc_ut1(
    epoch: Annotated[str, typer.Argument(..., help="Epoch-like to get EOP data for")],
    product: Annotated[
        ProductType, typer.Option(help="Type of EOP data to retrieve")
    ] = ProductType.standard,
    source: Annotated[
        ProductSource, typer.Option(help="Source of EOP data")
    ] = ProductSource.default,
    filepath: Annotated[
        Path, typer.Option(help="Filepath to EOP data. Only used if source is file")
    ] = None,
):
    # Set EOP Data
    eop = get_global_eop_source(product, source, filepath)
    brahe.set_global_eop_provider(eop)

    epc = brahe.Epoch(epoch)
    try:
        typer.echo(brahe.get_global_ut1_utc(epc.mjd()))
    except OSError:
        typer.echo(f"Error: Input epoch {epoch} is out of range for EOP data.")
        raise typer.Exit(code=1)


@app.command()
def get_polar_motion(
    epoch: Annotated[str, typer.Argument(..., help="Epoch-like to get EOP data for")],
    product: Annotated[
        ProductType, typer.Option(help="Type of EOP data to retrieve")
    ] = ProductType.standard,
    source: Annotated[
        ProductSource, typer.Option(help="Source of EOP data")
    ] = ProductSource.default,
    filepath: Annotated[
        Path, typer.Option(help="Filepath to EOP data. Only used if source is file")
    ] = None,
):
    # Set EOP Data
    eop = get_global_eop_source(product, source, filepath)
    brahe.set_global_eop_provider(eop)

    epc = brahe.Epoch(epoch)
    try:
        pm_x, pm_y = brahe.get_global_pm(epc.mjd())
        typer.echo(f"{pm_x}, {pm_y}")
    except OSError:
        typer.echo(f"Error: Input epoch {epoch} is out of range for EOP data.")
        raise typer.Exit(code=1)


@app.command()
def get_cip_offset(
    epoch: Annotated[str, typer.Argument(..., help="Epoch-like to get EOP data for")],
    product: Annotated[
        ProductType, typer.Option(help="Type of EOP data to retrieve")
    ] = ProductType.standard,
    source: Annotated[
        ProductSource, typer.Option(help="Source of EOP data")
    ] = ProductSource.default,
    filepath: Annotated[
        Path, typer.Option(help="Filepath to EOP data. Only used if source is file")
    ] = None,
):
    # Set EOP Data
    eop = get_global_eop_source(product, source, filepath)
    brahe.set_global_eop_provider(eop)

    epc = brahe.Epoch(epoch)
    try:
        dX, dY = brahe.get_global_dxdy(epc.mjd())
        typer.echo(f"{dX}, {dY}")
    except OSError:
        typer.echo(f"Error: Input epoch {epoch} is out of range for EOP data.")
        raise typer.Exit(code=1)


@app.command()
def get_lod(
    epoch: Annotated[str, typer.Argument(..., help="Epoch-like to get EOP data for")],
    product: Annotated[
        ProductType, typer.Option(help="Type of EOP data to retrieve")
    ] = ProductType.standard,
    source: Annotated[
        ProductSource, typer.Option(help="Source of EOP data")
    ] = ProductSource.default,
    filepath: Annotated[
        Path, typer.Option(help="Filepath to EOP data. Only used if source is file")
    ] = None,
):
    # Set EOP Data
    eop = get_global_eop_source(product, source, filepath)
    brahe.set_global_eop_provider(eop)

    epc = brahe.Epoch(epoch)
    try:
        typer.echo(brahe.get_global_lod(epc.mjd()))
    except OSError:
        typer.echo(f"Error: Input epoch {epoch} is out of range for EOP data.")
        raise typer.Exit(code=1)
