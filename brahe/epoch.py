# -*- coding: utf-8 -*-
"""The epoch module provides the ``Epoch`` class and associated helper methods.

Note:
    Time system transformations calls rely on the `pysofa2 <https://github.com/duncaneddy/pysofa2/>`_
    module to provide fast and accurate conversions. The pysofa2 also handles
    insertions of leap seconds.
"""

# Imports
import logging
import enum
import re       as _re
import datetime as _datetime
import copy     as copy
import pysofa2  as _sofa
import math     as math
import typing   as typing

# Brahe Imports
from   brahe.utils       import logger
import brahe.constants   as _constants
import brahe.time        as _bhtime

#############
# Constants #
#############

# TODO: Eventually implement Enum-Based Time SYstems
# class TimeSystem(enum.Enum):
#     '''Valid time systems.
#     '''
#     GPS = 0
#     TAI = 1
#     TT  = 2
#     UTC = 3
#     UT1 = 4

#     @staticmethod
#     def list():
#         return list(map(lambda tsys: tsys.value, TimeSystem))

VALID_TIME_SYSTEMS = ["GPS", "TAI", "TT", "UTC", "UT1"]

VALID_EPOCH_REGEX = [
    r"^(\d{4})\-(\d{2})\-(\d{2})$",
    r"^(\d{4})\-(\d{2})\-(\d{2})[T](\d{2})\:(\d{2})\:(\d{2})[Z]$",
    r"^(\d{4})\-(\d{2})\-(\d{2})[T](\d{2})\:(\d{2})\:(\d{2})[.](\d*)[Z]$",
    r"^(\d{4})(\d{2})(\d{2})[T](\d{2})(\d{2})(\d{2})[Z]$",
    r"^(\d{4})\-(\d{2})\-(\d{2})\s(\d{2})\:(\d{2})\:(\d{2})\.*\s*(\d*)\s*([A-Z]*)$"
]

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

def _epoch_to_jdfd(epc, tsys:str="UTC"):
    """Compute the two-part date format used by SOFA.jl functions forr a given Epoch.

    Args:
        epc (Epoch): Epoch
        tsys (str): Time system for output

    Returns:
        d1 (float): First part of two part date. Units: *days*
        d2 (float): Second part of two part date. Units: *days*
    """
    
    jd, fd = epc.days, (epc.seconds + epc.nanoseconds/1.0e9)/86400.0

    offset = _bhtime.time_system_offset(jd, fd, "TAI", tsys)

    return epc.days, (epc.seconds + offset + epc.nanoseconds/1.0e9)/86400.0

def _align_epoch_data(days:int, seconds:int, nanoseconds:float):
    # Align nanoseconds to [0, 1.0e9)
    
    while nanoseconds < 0.0:
        n_ns = math.fabs(nanoseconds)//1.0e9
        nanoseconds += 1.0e9
        seconds     -= 1

    while nanoseconds >= 1.0e9:
        nanoseconds -= 1.0e9
        seconds     += 1

    # Align seconds to [0, 86400)
    if seconds < 0:
        seconds += 86400
        days    -= 1

    if seconds >= 86400:
        seconds -= 86400
        days    += 1

    return days, seconds, nanoseconds

def _valid_epoch_string(string):
    """Checks whether a given string is a valid Epoch string.
    """

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
        tsys (str): Time system this Epoch was initialized in.
        days (int): Elapsed days since 0 JD TAI
        seconds (int): Total seconds into day
        nanseconds (float): Nanoseconds 

    Note:
        days, seconds, and nanoseconds is stored internally in the TAI reference
        systems. These values should not be accessed directly, but instead accessed
        through ``Epoch`` class methods which transform the state member date into
        the appropricate time system and format.
    """

    def __init__(self, *args, tsys='UTC', **kwargs):
        """Initialize Epoch Class. Supports multiple valid constructors.

        Constructors:
            Epoch(2018, 1, 1)
            Epoch(2018, 1, 1, 12, 0, 0, 0)
            Epoch(2018, 1, 1, 12, 0, 0, 0, tsys="GPS")
            Epoch("2018-01-01T12:00:00Z")
            Epoch(datetime.datetime(2018, 1, 1, 12, 0, 0, 0))
            Epoch(Epoch(2018, 1, 1, 12, 0, 0, 0))
        """

        # Initialize time system
        if not valid_time_system(tsys):
            raise RuntimeError('Invalid time system %s' % tsys)
        
        # Initialize Basic data members
        self.tsys        = tsys
        self.days        = 0
        self.seconds     = 0
        self.nanoseconds = 0.0

        # Handle args initialization
        if len(args) == 1:
            if isinstance(args[0], str):
                self._init_string(args[0], tsys=tsys)
            elif isinstance(args[0], _datetime.datetime):
                self._init_datetime(args[0], tsys=tsys)
            elif isinstance(args[0], Epoch):
                self._init_epoch(args[0])
        elif 3 <= len(args) <= 7:
            self._init_date(*args, tsys=tsys)
        else:
            raise RuntimeError('No inputs provided for Epoch initialization')
        
        # TODO: Hangle kwargs initialization

    def _init_date(self, year:int, month:int, day:int=0, 
            hour:int=0, minute:int=0, second:float=0.0, nanosecond=0.0, 
            tsys:str="UTC"):
        """Initialize Epoch from calendar date.

        Args:
            year (int): Year
            month (int): Month
            day (int): Day
            hour (int): Hour
            minute (int): Minute
            second (float): Seconds
            nanoseconds (float): Nanoseconds
        """

        jd, fd = _sofa.Dtf2d("TAI", year, month, day, hour, minute, 0)
        
        # Get time system offset based on days and fractional days using SOFA
        tsys_offset     = _bhtime.time_system_offset(jd, fd, tsys, "TAI")
        foffset, offset = divmod(tsys_offset, 1.0)

        # Ensure days is an integer number, add entire fractional component to the
        # fractional days variable
        days, fdays = divmod(jd, 1.0)
        fd          = fd + fdays

        # Convert fractional days into total seconds still retaining fractional part
        seconds = fd * 86400.0
        secs, fsecs = divmod(seconds, 1.0)

        # Now trip second of the fractional part
        second, fsecond = divmod(second, 1.0)

        # Add the integer parts together
        seconds = secs + second + offset

        # Convert the fractional parts to nanoseconds
        nanoseconds = nanosecond + fsecs*1.0e9 + fsecond*1.0e9 + foffset*1.0e9
        
        # Ensure type consistency of input:
        days        = int(days)
        seconds     = int(seconds)
        nanoseconds = float(nanoseconds)

        self.tsys = tsys
        self.days, self.seconds, self.nanoseconds = _align_epoch_data(days, seconds, nanoseconds)

    def _init_string(self, string:str, tsys:str="UTC"):
        """Initialize Epoch instance from a string.

        Args:
            datestring (str): str object to initialize from
        """
        year       = 0
        month      = 0
        day        = 0
        hour       = 0
        minute     = 0
        second     = 0.0
        nanosecond = 0.0
        tsys       = "UTC"

        m = None
        # Iterate through valid regex string 
        for regex in VALID_EPOCH_REGEX:
            m = _re.match(regex, string)
            if  m != None:
                # Parse date (common to all)
                year  = int(m.group(1))
                month = int(m.group(2))
                day   = int(m.group(3))

                # Parse time (most have this)
                if len(m.groups()) >= 6:
                    hour   = int(m.group(4))
                    minute = int(m.group(5))
                    second = float(m.group(6))

                # Parse additional types
                if len(m.groups()) == 7:
                    nanosecond = float(m.group(7))*10**(9-len(m.group(7)))
                elif len(m.groups()) == 8:
                    if m[7] != "":
                        nanosecond = float(m.group(7))*10**(9-len(m.group(7)))
                    
                    tsys = str(m.group(8))

                # Exit early since a match has been found
                break

        # No valid match found throw error
        if m == None:
            raise RuntimeError(f"Invalid Epoch string. \"{string}\" is not iso8061 compliant.")

        # Initialize from date.
        self._init_date(year, month, day, hour, minute, second, nanosecond, tsys=tsys)

    def _init_datetime(self, datetime:_datetime.datetime, tsys:str="UTC"):
        """Initialize Epoch instance from a datetime instance.

        Args:
            datetime (datetime.datetime): datetime object to initialize from
        """

        jd, fd = _sofa.Dtf2d("TAI", datetime.year, datetime.month, datetime.day, 
                                    datetime.hour, datetime.minute, 0)

        
        # Get time system offset based on days and fractional days using SOFA
        tsys_offset     = _bhtime.time_system_offset(jd, fd, tsys, "TAI")
        foffset, offset = divmod(tsys_offset, 1.0)


        # Ensure days is an integer number, add entire fractional component to the
        # fractional days variable
        days, fdays = divmod(jd, 1.0)
        fd          = fd + fdays

        # Convert fractional days into total seconds still retaining fractional part
        seconds = fd * 86400.0
        secs, fsecs = divmod(seconds, 1.0)

        # Now trip second of the fractional part
        second, fsecond = divmod(datetime.second + datetime.microsecond/1.0e6, 1.0)

        # Add the integer parts together
        seconds = secs + second + offset

        # Convert the fractional parts to nanoseconds
        nanoseconds = fsecs*1.0e9 + fsecond*1.0e9 + foffset*1.0e9
        
        # Ensure type consistency of input:
        days        = int(days)
        seconds     = int(seconds)
        nanoseconds = float(nanoseconds)


        self.days, self.seconds, self.nanoseconds = _align_epoch_data(days, seconds, nanoseconds)

    def _init_epoch(self, epc):
        """Initialize Epoch instance from another Epoch.

        Args:
            epc (Epoch): Epoch to copy state from
        """

        # Copy class variables
        self.tsys        = epc.tsys
        self.days        = epc.days
        self.seconds     = epc.seconds
        self.nanoseconds = epc.nanoseconds

    # Artithmetic
    def __iadd__(self, other):
        # Immidiately separate seconds and fractional seconds
        seconds, fseconds = divmod(other, 1.0)
        seconds = int(seconds) # Convert seconds to integer seconds

        # Compute time delta for each component
        dt_days        = seconds//86400
        dt_seconds     = seconds % 86400
        dt_nanoseconds = fseconds*1e9

        # Perform additon to get new epoch
        days        = self.days + dt_days
        seconds     = self.seconds + dt_seconds
        nanoseconds = self.nanoseconds + dt_nanoseconds

        # Align to proper ranges in reverse order
        self.days, self.seconds, self.nanoseconds = _align_epoch_data(days, seconds, nanoseconds)

        return self

    def __isub__(self, other):
        self += -other
        return self

    def __add__(self, other):
        # Create new epoch 
        epc  = copy.deepcopy(self)
        epc += other
        return epc

    def __sub__(self, other):
        if isinstance(other, Epoch):
            # Difference between two Epochs
            return (self.days - other.days)*86400.0 + \
                   (self.seconds - other.seconds) + \
                   (self.nanoseconds - other.nanoseconds)/1e9
        
        else:
            # Subtract seconds from Epoch
            epc  = copy.deepcopy(self)
            epc -= other
            return epc

    # Logical Operators
    def __eq__(self, other):
        return ((self.days == other.days) and
                (self.seconds == other.seconds) and
                (abs(self.nanoseconds - other.nanoseconds) < 10**-3))

    def __ne__(self, other):
        return ((self.days != other.days) or
            (self.seconds != other.seconds) or
            not (abs(self.nanoseconds - other.nanoseconds) < 10**-3))

    def __lt__(self, other):
        return ((self.days < other.days) or
            ((self.days == other.days) and
             (self.seconds < other.seconds)) or
             ((self.days == other.days) and
             (self.seconds == other.seconds) and
             (self.nanoseconds < other.nanoseconds)))

    def __le__(self, other):
        return ((self.days < other.days) or
            ((self.days == other.days) and
            (self.seconds < other.seconds)) or
            ((self.days == other.days) and
            (self.seconds == other.seconds) and
            (self.nanoseconds <= other.nanoseconds)))

    def __gt__(self, other):
        return ((self.days > other.days) or
            ((self.days == other.days) and
             (self.seconds > other.seconds)) or
             ((self.days == other.days) and
             (self.seconds == other.seconds) and
             (self.nanoseconds > other.nanoseconds)))

    def __ge__(self, other):
        return ((self.days > other.days) or 
            ((self.days == other.days) and
            (self.seconds > other.seconds)) or
            ((self.days == other.days) and
            (self.seconds == other.seconds) and
            (self.nanoseconds >= other.nanoseconds)))

    def caldate(self, tsys:typing.Optional[str]=None):
        """Return the Gregorian calendar date for a specific 

        Args:
            tsys (str): Time system to compute output in.

        Returns:
            year (int): Year of epoch
            month (int): Month of epoch
            day (int): Day of epoch
            hour (int): Hour of epoch
            minute (int): Minute of epoch
            second (float): Second of epoch
            nanoseconds (float): Year of epoch
        """

        # Use time system from initialization if none is provided
        if not tsys:
            tsys=self.tsys

        # Validate time system input
        if not valid_time_system(tsys):
            raise RuntimeError('Invalid time system %s' % tsys)

        # Get offset between input and 
        jd, fd = _epoch_to_jdfd(self, "TAI")
        offset = _bhtime.time_system_offset(jd, fd, "TAI", tsys)

        iy, im, id, ihmsf = _sofa.D2dtf(tsys, 9,
                                        self.days,
                                        (self.seconds + offset + self.nanoseconds/1.0e9)/86400.0)

        return iy, im, id, int(ihmsf[0]), int(ihmsf[1]), float(ihmsf[2]), float(ihmsf[3])

    def year(self, tsys:typing.Optional[str]=None):
        '''Year of epoch.

        Args:
            tsys (str, optional): Time system to produce output in

        Returns
            int: Year of epoch
        '''

        year, _, _, _, _, _, _ = self.caldate(tsys=tsys)

        return year

    def month(self, tsys:typing.Optional[str]=None):
        '''Month of epoch.

        Args:
            tsys (str, optional): Time system to produce output in

        Returns
            int: Month of epoch
        '''

        _, month, _, _, _, _, _ = self.caldate(tsys=tsys)

        return month

    def day(self, tsys:typing.Optional[str]=None):
        '''Day of epoch.

        Args:
            tsys (str, optional): Time system to produce output in

        Returns
            int: Day of epoch
        '''

        _, _, day, _, _, _, _ = self.caldate(tsys=tsys)

        return day

    def hour(self, tsys:typing.Optional[str]=None):
        '''Hour of epoch.

        Args:
            tsys (str, optional): Time system to produce output in

        Returns
            int: Hour of epoch
        '''

        _, _, _, hour, _, _, _ = self.caldate(tsys=tsys)

        return hour

    def minute(self, tsys:typing.Optional[str]=None):
        '''Minute of epoch.

        Args:
            tsys (str, optional): Time system to produce output in

        Returns
            int: Minute of epoch
        '''

        _, _, _, _, minute, _, _ = self.caldate(tsys=tsys)

        return minute

    def second(self, tsys:typing.Optional[str]=None):
        '''Second of epoch.

        Args:
            tsys (str, optional): Time system to produce output in

        Returns
            float: Second of epoch
        '''

        _, _, _, _, _, second, nanoseconds = self.caldate(tsys=tsys)

        return second + nanoseconds*1.0e9

    def gast(self, use_degrees:bool=False):
        """Compute the Greenwich Apparent Sidereal Time for the given Epoch.

        Args:
            use_degrees (bool): Return output in degrees (Default: false)

        Returns:
            gast (float): Greenwich Mean Sidereal Time [rad/deg]
        """
        
        uta, utb = _epoch_to_jdfd(self, tsys="UT1")
        tta, ttb = _epoch_to_jdfd(self, tsys="TT")

        gast = _sofa.Gst06a(uta, utb, tta, ttb)

        return gast*180.0/math.pi if use_degrees else gast

    def gmst(self, use_degrees:bool=False):
        """Compute the Greenwich Mean Sidereal Time for the given Epoch.

        Args:
            use_degrees (bool): Return output in degrees (Default: false)

        Returns:
            gmst (float): Greenwich Mean Sidereal Time [rad/deg]
        """

        uta, utb = _epoch_to_jdfd(self, tsys="UT1")
        tta, ttb = _epoch_to_jdfd(self, tsys="TT")

        gmst = _sofa.Gmst06(uta, utb, tta, ttb)
        
        return gmst*180.0/math.pi if use_degrees else gmst

    def mjd(self, tsys:typing.Optional[str]=None):
        """Return Epoch as a modified Julian date

        Args:
            tsys (str): time system to provide output in.

        Returns:
            mjd (float): Julian date of the epoch in the requested time system
        """

        # Use time system from initialization if none is provided
        if not tsys:
            tsys=self.tsys

        # Validate time system input
        if not valid_time_system(tsys):
            raise RuntimeError('Invalid time system %s' % tsys)

        jd, fd = _epoch_to_jdfd(self, tsys=tsys)

        return (jd - _constants.MJD_ZERO) + fd

    def jd(self, tsys:typing.Optional[str]=None):
        """Compute the Julian Date for a specific epoch

        Args:
            tsys (str): time system to provide output in.

        Returns:
            jd (float): Julian date of the epoch in the requested time system
        """

        # Use time system from initialization if none is provided
        if not tsys:
            tsys=self.tsys

        # Validate time system input
        if not valid_time_system(tsys):
            raise RuntimeError('Invalid time system %s' % tsys)
        
        jd, fd = _epoch_to_jdfd(self, tsys=tsys)

        return jd + fd

    def day_of_year(self, tsys:typing.Optional[str]=None):
        """Return the day-of-year number for a given `Epoch`. 

        January 1 0h of each year will return 1.

        Args:
            tsys (str): time system to provide output in.

        Returns:
            doy (float): Day of year number. 
        """
        
        # Use time system from initialization if none is provided
        if not tsys:
            tsys=self.tsys

        # Validate time system input
        if not valid_time_system(tsys):
            raise RuntimeError('Invalid time system %s' % tsys)

         # Compute MJD of first day of yearr
        date = self.caldate(tsys=tsys)
        mjd0 = _bhtime.caldate_to_mjd(date[0], 1, 1)

        # Compute MJD of current day
        jd, fd = _epoch_to_jdfd(self, tsys=tsys)
        mjd = (jd - _constants.MJD_ZERO) + fd
    
        # Get day of year is the difference
        doy = mjd - mjd0 + 1.0

        return doy

    def to_datetime(self, tsys:typing.Optional[str]=None):
        """Return equivalent datetime object for time system

        Args:
            tsys (str): time system to provide output in.

        Returns:
            dt (datetime): Datetime object storing epoch informatiotn.
        """
        # Use time system from initialization if none is provided
        if not tsys:
            tsys=self.tsys

        # Validate time system input
        if not valid_time_system(tsys):
            raise RuntimeError('Invalid time system %s' % tsys)

        year, month, day, hour, minute, second, nanosecond = self.caldate(tsys=tsys)

        return _datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), math.floor(nanosecond/1e3))

    def isoformat(self, tsys:str="UTC"):
        '''Return date and time as an ISO 8061 compliant string.
        '''
        return '%4d-%02d-%02dT%02d:%02d:%02.0fZ' % self.caldate('UTC')[0:6]

    def __str__(self, tsys:typing.Optional[str]=None):
        year, month, day, hour, minute, second, microsecond = self.caldate()
        second += microsecond/1.0e6
        return '%4d-%02d-%02d %02d:%02d:%06.3f %s' % (year, month, day, hour, minute, second, self.tsys)

    def __repr__(self):
        return f'<Epoch {hex(id(self))}: {self.days}-{self.seconds}-{self.nanoseconds}>'

    def __hash__(self):
        return hash(tuple((self.days, self.seconds, self.nanoseconds)))

def epoch_range(epoch_start, epoch_end, step=1.0):
    '''Generator that provides epochs at evenly spaced intervals between the
    start (inclusive) and stop (inclusive) times.

    Args:
        epoch_start (:obj:`Epoch`): Start epoch of range
        epoch_end (:obj:`Epoch`): End epoch of range
        step (float): Time increment in seconds.

    Returns:
        epoch (:obj:`Epoch`): Sequential epoch in range from epoch_start to
            epoch_end, inclusive.
    '''

    # Make sure step size is non-zero
    if math.fabs(step) == 0:
        raise ValueError('A positve step size is required.')

    # Make stepsize positive
    step = math.fabs(step)

    # Create running step
    epc = Epoch(epoch_start)

    # Iterate over all times in range creating them
    h = 0

    if epoch_end > epoch_start:
        while epc < epoch_end:
            yield Epoch(epc)
            h = min(step, epoch_end - epc)
            epc += h
    else:
        while epc > epoch_end:
            yield Epoch(epc)
            h = min(step, epc - epoch_end)
            epc -= h

    # If a step significantly different from the nominal step-size was taken
    # as the last step, output one more epoch. This allows for non-integer
    # divisible ranges, while still preventing duplicates of the final step.
    if math.fabs(h) > 1.0e-10:
        yield Epoch(epc)