# -*- coding: utf-8 -*-
"""The eop module provides data structures used to interact and access 
Earth-releated data files throughout the module.
"""
import logging
import math
import typing
import numpy as np

# Package imports
from brahe.utils import logger
from brahe.constants import DATA_PATH, AS2RAD

# Constants
IERS_AB_EOP_DATA = DATA_PATH / 'iau2000A_finals_ab.txt'
"""Path of IERS Finals AB data provided by module.
"""

IERS_C04_DATA = DATA_PATH / 'iau2000A_c04_14.txt'
"""Path of IERS C04 soultion data provided by module.
"""

DEFAULT_EOP_DATA = IERS_AB_EOP_DATA
"""Path of default earth orientation file used by the module. This is used
by the Epoch class compute offsets between various time systems.
"""

#########
# Utils #
#########

# Helper to read C04 data file into a dictionary
def _read_c04_file(filepath:str) -> None:
    """Read IERS C04-formatted Earth Orientation Parameter (EOP) data file.

    Args:
        filepath (str): Path to data file

    Returns:
        data (dict): Dictionary of (ut1-utc, xp, yp) keyed to the MJD utc of the data.
    """
    data = {}

    with open(filepath) as input_file:
        for _ in range(14):
            input_file.readline()

        for line in input_file:
            dat     = tuple(line.strip().split())
            mjd_utc = int(dat[3])             # MJD (UTC)
            ut1_utc = float(dat[6])           # UT1-UTC [s]
            xp      = float(dat[4])*AS2RAD   # xp [rad]
            yp      = float(dat[5])*AS2RAD   # yp [rad]
            data.update({mjd_utc:(ut1_utc, xp, yp)}) 

    return data

# Helper to read IERS Bulletin A/B data file into a dictionary
def _read_2000ab_file(filepath:str) -> None:
    """Read IERS Buelletin A/B-formatted Earth Orientation Parameter (EOP) data file.

    Args:
        filepath (str): Path to data file

    Returns:
        data (dict): Dictionary of (ut1-utc, xp, yp) keyed to the MJD utc of the data.
    """
    data = {}

    with open(filepath) as input_file:
        for line in input_file:
            if line[16:17] in ['P', 'I']:
                mjd_utc = int(line[7:12])             # MJD (UTC)
                ut1_utc = float(line[58:68])          # UT1-UTC [s]
                xp      = float(line[18:27])*AS2RAD  # xp [rad]
                yp      = float(line[37:46])*AS2RAD  # yp [rad]
                data.update({mjd_utc:(ut1_utc, xp, yp)})

    return data

################################
# Earth Orientation Parameters #
################################

class EOP():
    """Class to store Earth Orientation parameters and data.
    """

    # Class data memeters
    _initialized = False
    _data = {}

    @classmethod
    def load(cls, filepath:str=DEFAULT_EOP_DATA) -> None:
        """Load Earth orientation data into class memory.

        Args:
            filepath (str): Path to Earth orientation data file to load and use
                in module.
        """

        # Detect input file format type (C04 or A/B Bulletin)
        is_c04 = False

        # Detect file type
        with open(filepath) as input_file:
            # Read to C04 text flag
            for _ in range(4):
                input_file.readline().strip() # Advance file read 4 liens

            line = input_file.readline().strip()

            if line[-3:] == 'C04':
                is_c04 = True

        # Load data file
        if is_c04:
            cls._data = _read_c04_file(filepath)
        else:
            cls._data = _read_2000ab_file(filepath)

        # Flag initialization
        cls._initialized = True

    @classmethod
    def clear(cls) -> None:
        """Clear loaded EOP data
        """
        cls._initialized = False
        cls._data.clear()

    @classmethod
    def set(cls, mjd_utc:int, ut1_utc:float, xp:float, yp:float) -> None:
        """Insert or set Earth orientation values for specified date.

        Args:
            mjd_utc (int): Modified julian date of data Must be aligned to 0h 
                UTC. Units: *seconds*
            ut1_utc (float): UT1-UTC offset. Units: *arcseconds*
            xp (float): UT1-UTC offset Units: *arcseconds*
        """

        cls._data[int(mjd_utc)] = (ut1_utc, xp*AS2RAD, yp*AS2RAD)


    @classmethod
    def initialized(cls) -> None:
        """Return whether data has been initialized
        """
        return cls._initialized

    @classmethod
    def _initialize(cls) -> None:
        """Initialize class (load Earth Orientation Data) if it has not already
        been done.
        """

        if not cls._initialized:
            logger.warning('No Earth orientation data loaded. Loading default file: %s' % (DEFAULT_EOP_FILE))
            cls.load(filepath=DEFAULT_EOP_DATA)

    @classmethod
    def eop(cls, mjd_utc:float, interp:bool=False) -> typing.Tuple[float, float, float]:
        """Return the specified Earth orientation parameters based on.

        Args:
            mjd_utc (float): Modified Julian Date in the UTC time system.

        Returns:
            ut1_utc (float): UT1 - UTC time system offset. Units: *seconds*
            xp (float): x-vector component of polar offset. Units: *radians*
            yp (float): x-vector component of polar offset. Units: *radians*
        """

        # Ensure class is initialized
        cls._initialize()

        if interp:
            # Linearly interpolate output to time
            x1 = math.floor(mjd_utc)
            x2 = math.floor(mjd_utc) + 1

            # Get values converted to array for interpolation
            y1 = np.array([v for v in EOP._data[int(math.floor(mjd_utc))]])
            y2 = np.array([v for v in EOP._data[int(math.floor(mjd_utc)+1)]])
            x  = (y2 - y1)/(x2 - x1) * (mjd_utc - x1) + y1

            return x
        else:
            # Return Earth Orientation Parameters
            return cls._data[int(math.floor(mjd_utc))]

    @classmethod
    def pole_locator(cls, mjd_utc:float, interp:bool=False) -> typing.Tuple[float, float]:
        """Returns the angular location of Earth rotational axis.

        Args:
            mjd_utc (float): Modified Julian Date in the UTC time system.

        Returns:
            xp (float): x-vector component of polar offset. Units: *radians*
            yp (float): y-vector component of polar offset. Units: *radians*
        """

        # Ensure class is initialized
        cls._initialize()

        if interp:
            # Linearly interpolate output to time
            x1 = math.floor(mjd_utc)
            x2 = math.floor(mjd_utc) + 1

            # Get values converted to array for interpolation
            y1 = np.array(EOP._data[int(math.floor(mjd_utc))][1:3])
            y2 = np.array(EOP._data[int(math.floor(mjd_utc)+1)][1:3])
            x  = (y2 - y1)/(x2 - x1) * (mjd_utc - x1) + y1

            return x
        else:
            # Return Earth Orientation Parameters
            return cls._data[int(math.floor(mjd_utc))][1], cls._data[int(math.floor(mjd_utc))][2]

    @classmethod
    def xp(cls, mjd_utc:float, interp:bool=False) -> float:
        """Return the specified x-component of Earth Orientation .

        Args:
            mjd_utc (float): Modified Julian Date in the UTC time system.

        Returns:
            xp (float): x-vector component of polar offset. Units: *radians*
        """

        # Ensure class is initialized
        cls._initialize()

        if interp:
            # Linearly interpolate output to time
            x1 = math.floor(mjd_utc)
            x2 = math.floor(mjd_utc) + 1

            # Get values converted to array for interpolation
            y1 = EOP._data[int(math.floor(mjd_utc))][1]
            y2 = EOP._data[int(math.floor(mjd_utc)+1)][1]
            x  = (y2 - y1)/(x2 - x1) * (mjd_utc - x1) + y1

            return x
        else:
            # Return Earth Orientation Parameters
            return cls._data[int(math.floor(mjd_utc))][1]

    @classmethod
    def yp(cls, mjd_utc:float, interp:bool=False) -> float:
        """Return the specified y-component of Earth Orientation.

        Args:
            mjd_utc (float): Modified Julian Date in the UTC time system.

        Returns:
            yp (float): y-vector component of polar offset. Units: *radians*
        """

        # Ensure class is initialized
        cls._initialize()

        if interp:
            # Linearly interpolate output to time
            x1 = math.floor(mjd_utc)
            x2 = math.floor(mjd_utc) + 1

            # Get values converted to array for interpolation
            y1 = EOP._data[int(math.floor(mjd_utc))][2]
            y2 = EOP._data[int(math.floor(mjd_utc)+1)][2]
            x  = (y2 - y1)/(x2 - x1) * (mjd_utc - x1) + y1

            return x
        else:
            # Return Earth Orientation Parameters
            return cls._data[int(math.floor(mjd_utc))][2]

    @classmethod
    def ut1_utc(cls, mjd_utc:float, interp:bool=False) -> float:
        """Return the specified UT1 - UTC offset in seconds.

        Args:
            mjd_utc (float): Modified Julian Date in the UTC time system.

        Returns:
            ut1_utc (float): UT1 - UTC time system offset. Units: *seconds*
        """

        # Ensure class is initialized
        cls._initialize()

        if interp:
            # Linearly interpolate output to time
            x1 = math.floor(mjd_utc)
            x2 = math.floor(mjd_utc) + 1

            # Get values converted to array for interpolation
            y1 = EOP._data[int(math.floor(mjd_utc))][0]
            y2 = EOP._data[int(math.floor(mjd_utc)+1)][0]
            x  = (y2 - y1)/(x2 - x1) * (mjd_utc - x1) + y1

            return x
        else:
            # Return Earth Orientation Parameters
            return cls._data[int(math.floor(mjd_utc))][0]

    @classmethod
    def utc_ut1(cls, mjd_utc:float, interp:bool=False) -> float:
        """Return the specified UT1 - UTC offset in seconds.

        Args:
            mjd_utc (float): Modified Julian Date in the UTC time system.

        Returns:
            utc_ut1 (float): UT1 - UTC time system offset. Units: *seconds*
        """

        return -cls.ut1_utc(mjd_utc, interp=interp)