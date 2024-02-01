from enum import Enum
import typer
import brahe

app = typer.Typer()

@app.command()
def convert():
    typer.echo("Hello")