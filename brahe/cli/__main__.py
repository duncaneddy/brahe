from typing import Annotated

import typer

import brahe.cli.celestrak as celestrak_cli
import brahe.cli.spacetrack as spacetrack_cli
import brahe.logging
from brahe.cli import access, datasets, eop, orbits, time, transform

app = typer.Typer(name="brahe")
app.add_typer(eop.app, name="eop")
app.add_typer(time.app, name="time")
app.add_typer(orbits.app, name="orbits")
app.add_typer(transform.app, name="transform")
app.add_typer(access.app, name="access")
app.add_typer(datasets.app, name="datasets")
app.add_typer(celestrak_cli.app, name="celestrak")
app.add_typer(spacetrack_cli.app, name="spacetrack")


@app.callback()
def main_callback(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output (INFO level)")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Enable debug output (DEBUG level)")
    ] = False,
):
    """
    Brahe - Satellite Dynamics and Astrodynamics CLI

    A command-line interface for orbital mechanics, time systems, and satellite operations.
    """
    # Set up logging based on flags
    brahe.logging.setup_cli_logging(verbose=verbose, debug=debug)


# Call the application (used by setup.py to create the entry hook)
def main():
    app()
