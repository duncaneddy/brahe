import typer
import brahe.cli.eop as eop
import brahe.cli.time as time
import brahe.cli.orbits as orbits
import brahe.cli.conversions as conversions
import brahe.cli.space_track as space_track
import brahe.cli.access as access

app = typer.Typer(name="brahe")
app.add_typer(eop.app, name="eop")
app.add_typer(time.app, name="time")
app.add_typer(orbits.app, name="orbits")
app.add_typer(conversions.app, name="conversions")
app.add_typer(space_track.app, name="space-track")
app.add_typer(access.app, name="access")

# Call the application (used by setup.py to create the entry hook)
def main():
    app()
