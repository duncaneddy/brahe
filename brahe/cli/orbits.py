from enum import Enum
import typer
import brahe

app = typer.Typer()

@app.command()
def orbital_period():
    typer.echo("Hello")

@app.command()
def anomaly_conversion():
    typer.echo("Hello")

@app.command()
def sun_sync_inclination():
    typer.echo("Hello")

@app.command()
def perigee_velocity():
    typer.echo("Hello")

@app.command()
def apogee_velocity():
    typer.echo("Hello")

@app.command()
def propagate():
    typer.echo("Hello")

@app.command()
def interpolate():
    typer.echo("Hello")