import typer
import brahe.cli.eop as eop
import brahe.cli.time as time
import brahe.cli.orbits as orbits
import brahe.cli.convert as convert
import brahe.cli.access as access
import brahe.cli.datasets as datasets

app = typer.Typer(name="brahe")
app.add_typer(eop.app, name="eop")
app.add_typer(time.app, name="time")
app.add_typer(orbits.app, name="orbits")
app.add_typer(convert.app, name="convert")
app.add_typer(access.app, name="access")
app.add_typer(datasets.app, name="datasets")


# Call the application (used by setup.py to create the entry hook)
def main():
    app()
