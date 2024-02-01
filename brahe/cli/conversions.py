from enum import Enum
import typer
import brahe

app = typer.Typer()

@app.command()
def frame():
    typer.echo("Hello")

@app.command()
def coordinates():
    typer.echo("Hello")

@app.command()
def attitude():
    typer.echo("Hello")