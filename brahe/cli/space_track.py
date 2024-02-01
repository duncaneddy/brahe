from enum import Enum
import typer
import brahe

app = typer.Typer()

@app.command()
def get_tle():
    typer.echo("Hello")