import click
from brahe.__about__ import __version__


# Define top level command group
@click.group()
@click.version_option(version=__version__)
def cli_group(**kwargs):
    pass


# Add Commands
from brahe.cli.update import update

cli_group.add_command(update)

if __name__ == '__main__':
    cli_group()