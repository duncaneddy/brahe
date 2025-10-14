import typer

app = typer.Typer()


@app.command()
def compute_contacts():
    typer.echo("Hello")
