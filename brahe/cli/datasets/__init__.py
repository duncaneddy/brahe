"""
CLI commands for datasets module
"""

import typer
from brahe.cli.datasets import groundstations

app = typer.Typer()
app.add_typer(groundstations.app, name="groundstations")
