# -*- coding: utf-8 -*-
"""This module provides function to download or update the data files used by
the brahe package.
"""

# Imports
import logging
import requests
import pathlib
import math
from typing import Optional

from brahe.constants import DATA_PATH

# Get Logger
logger = logging.getLogger(__name__)

################
# Data Sources #
################

URL_LS_DATA  = 'http://maia.usno.navy.mil/ser7/tai-utc.dat'
URL_IERS_AB  = 'https://datacenter.iers.org/data/latestVersion/9_FINALS.ALL_IAU2000_V2013_019.txt'
URL_IERS_C04 = 'https://datacenter.iers.org/data/latestVersion/224_EOP_C04_14.62-NOW.IAU2000A224.txt'

###############
# Data Update #
###############

def download_datafile(url:str, outdir:str='.', filename:Optional[str]=None):
    """Downloads datafile at URL endpoint to specified output directory

    Args:
        url (str): URL of data file resource 
        outdir (str): Path to output data directory. Default: `.`

    Note:
        Data source is http://maia.usno.navy.mil/ser7/tai-utc.dat
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
def download_leap_second_data(outdir:str='.'):
    """Download most recent leap second data file.

    Args:
        outdir (str): Path to output data directory. Default: `.`

    Note:
        Data source is http://maia.usno.navy.mil/ser7/tai-utc.dat
    """

    download_datafile(URL_LS_DATA, outdir=outdir, filename='tai_utc.txt')

# IERS Bulletin A and B - ITRF 2008 realization
def download_iers_bulletin_ab(outdir:str='.'): 
    """Download most recent IERS buelltin A/B data tables.

    Args:
        outdir (str): Path to output data directory. Default: `.`

    Note:
        Data source is https://datacenter.iers.org/data/latestVersion/9_FINALS.ALL_IAU2000_V2013_019.txt
    """

    download_datafile(URL_IERS_AB, outdir=outdir, filename='iau2000A_finals_ab.txt')

def download_iers_bulletin_c(outdir:str='.'):
    """Download most recent IERS bulletin C data tables.
    
    Args:
        outdir (str): Path to output data directory. Default: `.`

    Note:
        Data source is https://datacenter.iers.org/data/latestVersion/224_EOP_C04_14.62-NOW.IAU2000A224.txt
    """

    download_datafile(URL_IERS_C04, outdir=outdir, filename='iau2000A_c04_14.txt')

def download_all_data(outdir:str='.'):
    """Download all data files used by package.

    Args:
        outdir (str): Path to output data directory. Default: `.`
    """

    download_leap_second_data(outdir=outdir)
    download_iers_bulletin_ab(outdir=outdir)
    download_iers_bulletin_c(outdir=outdir)

def update_package_data():
    """Update all data files currently installed by the module.
    """
    
    # Default data path
    download_leap_second_data(DATA_PATH)
    download_iers_bulletin_ab(DATA_PATH)
    download_iers_bulletin_c(DATA_PATH)