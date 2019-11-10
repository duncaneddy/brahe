import click

# Data update data files
import brahe.constants as const
import brahe.utils as utils

@click.command()
@click.option('--type', default='all',
    type=click.Choice(['all', 'ab', 'iers', 'weather'], case_sensitive=False), help='Data file to update')
@click.option('--outdir', default=const.DATA_PATH,
    type=click.Path(), help='Output directory for updated data files. Can use this to specify an output directory other than the package default.')
def update(type, outdir):
    click.echo(f'Updating brahe package data.\nUpdate type selected: {type:s}')

    if type == 'ab':
        utils.download_iers_bulletin_ab(outdir)
    elif type == 'iers': 
        utils.download_leap_second_data(outdir)
        utils.download_iers_bulletin_ab(outdir)
        utils.download_iers_bulletin_c(outdir)
    elif type == 'weather':
        utils.download_spaceweather(outdir)
        utils.download_solarflux(outdir)
        utils.download_kpap(outdir)
    else:
        utils.download_all_data(outdir)