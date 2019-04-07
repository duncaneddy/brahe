# -*- coding: utf-8 -*-
"""The epoch module provides the ``Epoch`` class and associated helper methods.

Note:
    Time system transformations calls rely on the `pysofa2 <https://github.com/duncaneddy/pysofa2/>`_
    module to provide fast and accurate conversions. The pysofa2 also handles
    insertions of leap seconds.
"""

# Imports
import pysofa2           as _sofa
import datetime.datetime as datetime

import brahe.constants   as _constants
from brahe.earthmodels import EarthOrientationParameters as EOP

#############
# Constants #
#############

VALID_TIME_SYSTEMS = ["GPS", "TAI", "TT", "UTC", "UT1"]

###################
# Epoch Utilities #
###################

def valid_time_system(tsys:str):
    """Check if time system code is a valid time system identifier.

    Args:
        tsys (str): Time system

    Returns:
        valid (bool): Whether time system is valid or not
    """

    return tsys in VALID_TIME_SYSTEMS

def _convert_jd_fd(jd, fd, time_system='GPS'):
    '''Align internal class contents to internal 
    '''
    pass
    
    # # Ensure type consistency of input:
    # days        = Int(days)
    # seconds     = Int(seconds)
    # nanoseconds = Float64(nanoseconds)

    # days, seconds, nanoseconds = align_epoch_data(days, seconds, nanoseconds)

def _valid_epoch_string(string):
    '''Checks whether a given string is a valid Epoch string.
    '''

    if _re.match(r'^(\d{4})\-(\d{2})\-(\d{2})$', string):
        return True
    elif _re.match(r'^(\d{4})\-(\d{2})\-(\d{2})[T](\d{2})\:(\d{2})\:(\d{2})([+-])(\d{2})\:(\d{2})$', string):
        return True
    elif _re.match(r'^(\d{4})\-(\d{2})\-(\d{2})[T](\d{2})\:(\d{2})\:(\d{2})[Z]$', string):
        return True
    elif _re.match(r'^(\d{8})[T](\d{6})[Z]$', string):
        return True
    elif _re.match(r'^(\d{4})\-(\d{2})\-(\d{2})\s(\d{2})\:(\d{2})\:(\d{2})\.*\s*(\d*)\s*([A-Z]*)$', string):
        return True
    else:
        return False

###############
# Epoch Class #
###############

class Epoch():
    """The `Epoch` type represents a single instant in time. It is used throughout the
    SatelliteDynamics module. It is meant to provide a clear definition of moments
    in time and provide a convenient interface display time in various representations
    as well as in differrent time systems. The internal data members are also chosen
    such that the representation maintains nanosecond-precision in reprersenation
    of time and doesn't accumulate floating-point arithmetic errors larger than
    nanoseconds even after centuries.

    Attributes:
        days (int): Elapsed days since 0 JD TAI
        seconds (int): Total seconds into day
        nanseconds (float): Nanoseconds 
    """

    def __init__(self, year=None, month=None, day=None, hour:int=0, minute:int=0, second:int=0, nanosecond:float=0,
                 string:str=None, datetime:datetime=None, gps_week:int=None, gps_seconds:float=None, mjd:float=None, 
                 epoch:Epoch=None, tsys='UTC'):
        """Initialize Epoch Class. 
        """

        # Initialize time system
        if not valid_time_system(tsys):
            raise RuntimeError('Invalid time system %s' % tsys)
        self.tsys = tsys

        # Initialize Basic data members
        self.days        = 0
        self.seconds     = 0
        self.nanoseconds = 0.0

        # Detect other initial input and dispatch
        if isinstance(year, str):
            string   = year
        elif isinstance(year, datetime):
            datetime = year

        if all((year, month, day)):
            self._init_date(year, month, day, hour, minute, second, tsys=tsys)

    def _init_date(self, year:int, month:int, day:int, 
            hour:int, minute:int, second:float, tsys:str):
        """Initialize Epoch from date input
        """
        pass

    def _init_string(self, ):
        pass

    def _init_gps(self, ):
        pass

    def _init_posix(self, ):
        pass

    def _init_mjd(self, ):
        pass

    # Logical Operators
    def __eq__(self, other):
        pass

    def __ne__(self, other):
        pass

    def __lt__(self, other):
        pass

    def __le__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __ge__(self, other):
        pass