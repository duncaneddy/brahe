# -*- coding: utf-8 -*-
"""The time module provides various fundamental time system and time representation
conversions.

Note:
    Most function calls rely on the `pysofa2 <https://github.com/duncaneddy/pysofa2/>`_
    module to provide fast and accurate conversions. The pysofa2 also handles
    insertions of leap seconds.
"""

# Imports
import typing as _typing
import pysofa2 as _sofa

# Brahe Imports
from   brahe.utils import logger
import brahe.constants as _constants
from   brahe.eop import EOP as _EOP

################
# Time Methods #
################

def caldate_to_mjd(year:int, month:int, day:int, hour:int=0, minute:int=0, second:float=0.0, nanoseconds:float=0.0) -> float:
    """Convert a Gregorian calendar date to the equivalent Modified Julian Date 
    representation of that time instant.

    Args:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int): Hour
        minute (int): Minute
        second (float): Seconds
        nanoseconds (float): Nanoseconds

    Returns:
        mjd (float): Modified Julian Date of Epoch
    """

    jd, fd = _sofa.Dtf2d("TAI", year, month, day, hour, minute, second + nanoseconds/1.0e9)

    mjd = (jd - _constants.MJD_ZERO) + fd

    return mjd

def mjd_to_caldate(mjd:float) -> _typing.Tuple[float, float, float, float, float, float, float]:
    """Convert a Modified Julian Date to the equivalent Gregorian calendar date 
    representation of the same instant in time.

    Args:
        mjd (float): Modified Julian Date of Epoch

    Returns:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int): Hour
        minute (int): Minute
        second (float): Seconds
        nanoseconds (float): Nanoseconds
    """

    iy, im, id, ihmsf = _sofa.D2dtf("TAI", 9, _constants.MJD_ZERO, mjd)

    return iy, im, id, ihmsf[0], ihmsf[1], ihmsf[2], ihmsf[3]/1.0e9


def caldate_to_jd(year:int, month:int, day:int, hour:int=0, minute:int=0, second:float=0.0, nanoseconds:float=0.0) -> float:
    """Convert a Gregorian calendar date to the equivalent Julian Date 
    representation of that time instant.

    Args:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int): Hour
        minute (int): Minute
        second (float): Seconds
        nanoseconds (float): Nanoseconds

    Returns:
        jd (float): Julian Date of Epoch
    """

    jd, fd = _sofa.Dtf2d("TAI", year, month, day, hour, minute, second + nanoseconds/1.0e9)

    jd = jd + fd

    return jd

def jd_to_caldate(jd:float) -> _typing.Tuple[float, float, float, float, float, float, float]:
    """Convert a Julian Date to the equivalent Gregorian calendar date 
    representation of the same instant in time.

    Args:
        jd (float): Julian Date of Epoch

    Returns:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int): Hour
        minute (int): Minute
        second (float): Seconds
        nanoseconds (float): Nanoseconds
    """

    iy, im, id, ihmsf = _sofa.D2dtf("TAI", 9, jd, 0)

    return iy, im, id, ihmsf[0], ihmsf[1], ihmsf[2], ihmsf[3]/1.0e9

def time_system_offset(jd:float, fd:float, tsys_src:str, tsys_dest:str) -> float:
    """Compute the offset between two time systems at a given Epoch.

    The offset (in seconds) is computed as:

        time_system_offset = tsys_dest - tsys_src

    The value returned is the number of seconds that musted be added to the 
    source time system given the input epoch, to get the equivalent epoch.

    Conversions are accomplished using SOFA C library calls.

    Args:
        jd (float): Part 1 of two-part date (Julian days)
        fd (float): Part 2 of two-part date (Fractional days)
        tsys_src (str): Base time system
        tsys_dest (str): Destination time system

    Returns:
        offset (float): Offset between soruce and destination time systems in 
            seconds.
    """
    
    # If no transformation is needed needed return early
    if tsys_src == tsys_dest:
        return 0.0
    
    offset = 0.0

    # Convert To TAI 
    if tsys_src == "GPS":
        offset += _constants.TAI_GPS
    elif tsys_src == "TT":
        offset += _constants.TAI_TT
    elif tsys_src == "UTC":
        iy, im, id, ihmsf = _sofa.D2dtf("UTC", 6, jd, fd) # Returns TAI-UTC
        dutc = _sofa.Dat(iy, im, id, (ihmsf[0]*3600 + ihmsf[1]*60 + ihmsf[2] + ihmsf[3]/1e6)/86400.0)
        offset += dutc
    elif tsys_src == "UT1":
        # Convert UT1 -> UTC
        offset -= _EOP.ut1_utc((jd - _constants.MJD_ZERO) + fd)

        # Convert UTC -> TAI
        iy, im, id, ihmsf = _sofa.D2dtf("UTC", 6, jd, fd + offset) # Returns TAI-UTC
        dutc = _sofa.Dat(iy, im, id, (ihmsf[0]*3600 + ihmsf[1]*60 + ihmsf[2] + ihmsf[3]/1e6)/86400.0)
        offset += dutc
    elif tsys_src == "TAI":
        # Do nothing in this case
        pass

    # Covert from TAI to source
    if tsys_dest == "GPS":
        offset += _constants.GPS_TAI
    elif tsys_dest == "TT":
        offset += _constants.TT_TAI
    elif tsys_dest == "UTC":
        # Initial UTC guess
        u1, u2 = jd, fd + offset/86400.0

        # Iterate to get the UTC time
        for i in range(0, 3):
            d1, d2 = _sofa.Utctai(u1, u2)

            # Adjust UTC guess
            u1 += jd - d1
            u2 += fd + offset/86400.0 - d2

        # Compute Caldate from two-part date
        iy, im, id, ihmsf = _sofa.D2dtf("UTC", 6, u1, u2)

        dutc = _sofa.Dat(iy, im, id, (ihmsf[0]*3600 + ihmsf[1]*60 + ihmsf[2] + ihmsf[3]/1e6)/86400.0)
        offset -= dutc
    elif tsys_dest == "UT1":
        # Initial UTC guess
        u1, u2 = jd, fd + offset/86400.0

        # Iterate to get the UTC time
        for i in range(0, 3):
            d1, d2 = _sofa.Utctai(u1, u2)

            # Adjust UTC guess
            u1 += jd - d1
            u2 += fd + offset/86400.0 - d2

        # Compute Caldate from two-part date
        iy, im, id, ihmsf = _sofa.D2dtf("UTC", 6, u1, u2)

        dutc = _sofa.Dat(iy, im, id, (ihmsf[0]*3600 + ihmsf[1]*60 + ihmsf[2] + ihmsf[3]/1e6)/86400.0)
        offset -= dutc

        # Convert UTC to UT1
        offset += _EOP.ut1_utc(u1 + u2 + offset/86400.0 - _constants.MJD_ZERO)
    elif tsys_dest == "TAI":
        # Do nothing in this case
        pass

    return offset