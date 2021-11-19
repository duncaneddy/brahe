import logging
import requests
import urllib
import tempfile
import datetime
import typing
import pathlib
import numpy as np
import numba
from contextlib import closing

import brahe.constants as _const

# Setup logging
logger = logging.getLogger(__name__)

# Define common array-like type
AbstractArray = typing.NewType('AbstractArray', typing.Union[tuple, list, np.ndarray])

###############
# Mathematics #
###############

@numba.jit(nopython=True, cache=True)
def kron_delta(a:float, b:float) -> int:
    """Cannonical Kronecker Delta function.

    Returns 1 if inputs are equal returns 0 otherwise.

    Args:
        (:obj:`float`): First input argument
        (:obj:`float`): Second input argument 

    Returns:
        (:obj:`int`) Kronecker delta result {0, 1}
    """
    if a == b:
        return 1
    else:
        return 0

def fcross(a, b):
    '''Quickly compute cross-product between 3-length vectors a and b.

    Args:
        a (:obj:`np.ndarray`): First vector
        b (:obj:`np.ndarray`): Second vector

    Return:
        np.ndarray: Cross product between input vectors
    '''

    c1 = a[1]*b[2] - a[2]*b[1]
    c2 = a[2]*b[0] - a[0]*b[2]
    c3 = a[0]*b[1] - a[1]*b[0]

    return [c1, c2, c3]

################
# Data Sources #
################

URL_LS_DATA  = 'http://maia.usno.navy.mil/ser7/tai-utc.dat'
URL_IERS_AB  = 'https://datacenter.iers.org/data/latestVersion/9_FINALS.ALL_IAU2000_V2013_019.txt'
URL_IERS_C04 = 'https://datacenter.iers.org/data/latestVersion/224_EOP_C04_14.62-NOW.IAU2000A224.txt'
URL_CELESTRACK_SPACE_WEATHER = 'https://celestrak.com/SpaceData/sw19571001.txt'
URL_SOLAR_FLUX = 'ftp://ftp.seismo.nrcan.gc.ca/spaceweather/solar_flux/daily_flux_values/fluxtable.txt'
URL_KPAP_BASE = ''

###############
# Data Update #
###############

def download_datafile(url:str, outdir:str='.', filename:typing.Optional[str]=None) -> None:
    """Downloads datafile at URL endpoint to specified output directory

    Args:
        url (str): URL of data file resource 
        outdir (str): Path to output data directory. Default: `.`
    """

    # Get filepath to output file
    filename = filename if filename else url.split('/')[-1]
    filepath = pathlib.Path(outdir, filename)

    # GET request to retrive data
    response = requests.get(url)

    # write to file
    with open(filepath, "wb") as file:
        file.write(response.content)

# Get IERS leap second introductions
def download_leap_second_data(outdir:str='.') -> None:
    f"""Download most recent leap second data file.

    Args:
        outdir (str): Path to output data directory. Default: `.`

    Note:
        Data source is {URL_LS_DATA}
    """

    download_datafile(URL_LS_DATA, outdir=outdir, filename='tai_utc.txt')

# IERS Bulletin A and B - ITRF 2008 realization
def download_iers_bulletin_ab(outdir:str='.'): 
    f"""Download most recent IERS buelltin A/B data tables.

    Args:
        outdir (str): Path to output data directory. Default: `.`

    Note:
        Data source is {URL_IERS_AB}
    """

    download_datafile(URL_IERS_AB, outdir=outdir, filename='iau2000A_finals_ab.txt')

def download_iers_bulletin_c(outdir:str='.') -> None:
    f"""Download most recent IERS bulletin C data tables.
    
    Args:
        outdir (str): Path to output data directory. Default: `.`

    Note:
        Data source is {URL_IERS_C04}
    """

    download_datafile(URL_IERS_C04, outdir=outdir, filename='iau2000A_c04_14.txt')

def download_spaceweather(outdir:str='.') -> None:
    f"""Download most recent space weather data from Celestrak.
    
    Args:
        outdir (str): Path to output data directory. Default: `.`

    Note:
        Data source is {URL_CELESTRACK_SPACE_WEATHER}
    """

    download_datafile(URL_CELESTRACK_SPACE_WEATHER, outdir=outdir, filename='space_weather.txt')


def download_solarflux(outdir:str='.') -> None:
    f"""Download most recent solar flux data.
    
    Args:
        outdir (str): Path to output data directory. Default: `.`

    Note:
        Data source is {URL_SOLAR_FLUX}
    """

    with closing(urllib.request.urlopen(URL_SOLAR_FLUX)) as r:
        with open(pathlib.Path(outdir, 'fluxtable.txt'), 'wb') as fp:
            fp.write(r.read())

def download_kpap(outdir:str='.', start_year:int=2000, end_year:int=datetime.datetime.now().year) -> None:
    f"""Download most recent Geomagnetic (KP/AP) data.
    
    Args:
        outdir (str): Path to output data directory. Default: `.`

    Note:
        Data source is {URL_KPAP_BASE}
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download Files
        for year in range(start_year, end_year+1):
            fluxfile = f'kp{year}.wdc'
            logger.debug(f'Downloading {fluxfile}')
            with closing(urllib.request.urlopen(URL_KPAP_BASE + fluxfile)) as r:
                with open(pathlib.Path(tmpdir, fluxfile), 'wb') as fp:
                    fp.write(r.read())

        with open(pathlib.Path(outdir, "kpall.wdc"), 'w') as fp:
            for year in range(start_year, end_year+1):
                fluxfile = f'kp{year}.wdc'
                logger.debug(f'Merging {fluxfile}')
                with open(pathlib.Path(tmpdir, fluxfile), 'r') as fp_kpfile:
                    for line in fp_kpfile.readlines():
                        fp.write(line)

def download_all_data(outdir:str='.') -> None:
    """Download all data files used by package.

    Args:
        outdir (str): Path to output data directory. Default: `.`
    """

    # logger.info(f'Updating leap second data')
    # download_leap_second_data(outdir=outdir)
    
    logger.info(f'Updating IERS Bulletin A/B')
    download_iers_bulletin_ab(outdir=outdir)
    
    logger.info(f'Updating IERS Bulletin C')
    download_iers_bulletin_c(outdir=outdir)
    
    logger.info(f'Updating Clestrak space weather data')
    download_spaceweather(outdir=outdir)
    
    logger.info(f'Updating Penticon Solar Flux Data')
    download_solarflux(outdir=outdir)
    
    logger.info(f'Updating KP AP data')
    download_kpap(outdir=outdir)

def update_package_data() -> None:
    """Update all data files currently installed by the module.
    """
    
    # Default data path
    download_all_data(_const.DATA_PATH)