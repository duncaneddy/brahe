import typer

app = typer.Typer()


@app.command()
def get_tle():
    typer.echo("Hello")
