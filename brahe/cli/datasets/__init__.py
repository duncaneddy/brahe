"""
CLI commands for datasets module
"""

import typer
from brahe.cli.datasets import celestrak, groundstations

app = typer.Typer()
app.add_typer(celestrak.app, name="celestrak")
app.add_typer(groundstations.app, name="groundstations")
