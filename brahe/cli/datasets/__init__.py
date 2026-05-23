"""
CLI commands for datasets module
"""

import typer
from brahe.cli.datasets import groundstations, icgem

app = typer.Typer()
app.add_typer(groundstations.app, name="groundstations")
app.add_typer(icgem.app, name="icgem")
