from enum import Enum
import typer
import brahe

app = typer.Typer()

@app.command()
def compute_contacts():
    typer.echo("Hello")