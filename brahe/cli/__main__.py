import typer
from typing_extensions import Annotated
import brahe.cli.eop as eop
import brahe.cli.time as time
import brahe.cli.orbits as orbits
import brahe.cli.transform as transform
import brahe.cli.access as access
import brahe.cli.datasets as datasets
import brahe.logging

app = typer.Typer(name="brahe")
app.add_typer(eop.app, name="eop")
app.add_typer(time.app, name="time")
app.add_typer(orbits.app, name="orbits")
app.add_typer(transform.app, name="transform")
app.add_typer(access.app, name="access")
app.add_typer(datasets.app, name="datasets")


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
