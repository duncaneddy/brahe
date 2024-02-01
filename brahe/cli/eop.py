from enum import Enum
import typer
import brahe

class ProductType(str, Enum):
    standard = "standard"
    c04 = "c04"

app = typer.Typer()

@app.command()
def download(filepath:str = typer.Argument(..., help="Filepath to output data"),
             type:ProductType = typer.Option(..., help="Type of data product to download")):
    if type == ProductType.standard:
        brahe.download_standard_eop_file(filepath)
    else:
        brahe.download_c04_eop_file(filepath)