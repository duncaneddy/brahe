from enum import Enum
import typer
import brahe

class ProductType(str, Enum):
    standard = "standard"
    C04 = "C04"

app = typer.Typer()

@app.command()
def download(filepath:str = typer.Argument(..., help="Filepath to output data"),
             product:ProductType = typer.Option(..., help="Type of data product to download")):
    typer.echo("Hello")
    # if product == ProductType.standard:
    #     brahe.eop.download_standard_eop_file(filepath)
    # else:
    #     brahe.eop.download_c04_eop_file(filepath)