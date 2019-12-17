# -*- coding: utf-8 -*-
"""The tle module provides class definitions to provide an convenient interface
for interacting with NORAD Two-Line Element (TLE) sets, the associated SGP4
propagated, as well as how to convert the output from the base frame to other
common reference frames.

Note:
    The implementation of SGP4 propagator comes from Brandon Rhoade's Python
    implemntation `pysofa2 <https://github.com/duncaneddy/pysofa2/>`_. His
    python implementation is based on the original code provided by David
    Vallado in _Revisiting Spacetrack Report #3_
"""

# Imports
import math
import typing
import numpy as np
import sgp4.io
import sgp4.earth_gravity
import sgp4.propagation
import pysofa2 as _sofa

# Brahe Imports
from   brahe.utils import logger
import brahe.constants as _constants
import brahe.attitude as _att
import brahe.astro as _astro
import brahe.frames as _frames
from brahe.epoch import Epoch

#############
# TLE Class #
#############

def tle_format_exp(num:float) -> str:
    '''Format TLE as exponential.

    Args:
        num (float): Input number to format

    Returns:
        str: Number formated in TLE exponential format
    '''

    if num == 0.0:
        return ' 00000-0'

    # Get Exponent value 
    exp = math.floor(math.log10(math.fabs(num))) + 1

    # Get the number
    body = int(math.fabs(num) * 10**(-exp+5))

    # Construct string and return
    sign = '-' if num < 0 else ' '
    num_str = f'{sign:1s}{body:5d}-{int(math.fabs(exp)):1d}'

    return num_str

def tle_checksum(line:str) -> int:
    '''Compute checksum value for two-line element set line.

    Args:
        line (str): Input two line element 

    Returns:
        int: Checksum value.
    '''
        
    checksum = 0
    for c in line[0:68]:
        if c.isdigit():
            checksum += int(c)
        elif c == '-':
            checksum += 1

    checksum = checksum % 10

    return checksum

def validate_tle_line(line:str) -> bool:
    '''Validate if line is a valid TLE line.

    Args:
        line (str): Input two line element 

    Returns:
        bool: True if input string is a valid TLE line
    '''

    line_length = len(line)
    if line_length != 69:
        return False

    if int(line[68]) != tle_checksum(line):
        return False

    return True

def tle_gmst82(epoch:Epoch, use_degrees:bool=False):
    '''Compute Greenwich Mean Sidereal Time 1982 Model. Formulae taken from
    `Revisiting Spacetrack Report No 3` by David Vallado for use in transforming
    between the TEME and PEF frames.

    Args:
        epoch (:obj:`Epoch`): Epoch of transformation

    Returns:
        float: Greenwich mean sidereal time as angle. Units: Radians [0, 2pi)
        float: Rate of change of Greenwich mean sidereal time as angle. Units: Radians/second [0, 2pi)
    '''

    # Compute UT1 time
    jd_ut1 = epoch.jd(tsys='UT1')

    # jd_ut1 is days elapsed since January 1, 2000 12h UT1
    t = (jd_ut1 - 2451545.0) / 36525.0

    # Apply Formula C-1
    g  = 67310.54841 + (876000*3600 + 8640184.812866)*t + 0.093104*t**2 + -6.2*10**-6*t**3

    # Compute GMST as angle
    theta  = ((jd_ut1 % 1.0) + (g / 86400.0 % 1.0)) * 2*math.pi

    if use_degrees == True:
        theta *= 180.0/math.pi

    return theta

def tle_string_from_elements(epc:Epoch, oe:np.ndarray, norad_id:int=None,
        classification:str='U', intl_des:str='', input_sma:bool=False):
    '''Generate a TLE string from orbital elements.

    Args:
        epc (:obj:`Epoch`): Epoch of orbital elements
        oe (:obj:`np.ndarray`): Orbital Elements. Input expected to be:
            - oe[0] - n - Mean motion [rev/day]
            - oe[1] - e - Eccentricity [dimensionless]
            - oe[2] - i - Inclination [deg]
            - oe[3] - O - Right ascension [deg]
            - oe[4] - w - Argument of perigee [deg]
            - oe[5] - M - Mean anomaly [deg]
            - oe[6] - ndt2 - First derivative of mean motion divided by 2
            - oe[7] - nddt6 - Second derivative of mean motion divided by 6
            - oe[8] - bstar - B-star drag term
        norad_id (str): NORAD Catalog ID
        classification (str): Object classification. 'U' or 'C'
        intl_des (str): International Designator
        input_sma (bool): Interpert first argument input orbital elements as semi-major axis with units in meters.

    Returns:
        str: First line of orbital element set
        str: Second line of orbital element set
    '''

    if not norad_id:
        norad_id = ''

    if len(oe) != 9:
        raise RuntimeError('Input orbital elements must be of length 9.')

    # Format Line 1
    norad_id  = norad_id
    year      = epc.year('UTC') % 100
    doy       = epc.day_of_year('UTC')
    ndt2_sign = '-' if oe[6] < 0 else ' '
    ndt2      = ndt2_sign + f'{math.fabs(oe[6]):9.8f}'.lstrip('0')

    nddt6 = tle_format_exp(oe[7])
    bstar = tle_format_exp(oe[8])

    line1 = f'1 {norad_id:5d}{classification:1s} {intl_des:8s} ' + \
            f'{year:02d}{doy:12.8f} {ndt2:s} {nddt6:8s} ' + \
            f'{bstar:8s} 0 {0:4d}'

    line1_checksum = tle_checksum(line1)
    line1 += str(line1_checksum)

    # Format Line 2
    incl = oe[2]
    raan = oe[3]
    ecc = f'{oe[1]:9.7f}'.lstrip('0').lstrip('.') # Format eccentricity
    w = oe[4]
    M = oe[5]
    n = oe[0]

    # Convert Semi-major axis to rev/day
    if input_sma == True:
        a = oe[0]
        n = _astro.mean_motion(a)/(2.0*math.pi)*86400.0

    # Ensure alignment of values in degrees [0, 360)
    raan = math.fmod(raan, 360.0)
    w    = math.fmod(w, 360.0)
    M    = math.fmod(M, 360.0)

    # Construct line 2
    line2 = f'2 {norad_id:5d} {incl:8.4f} {raan:8.4f} {ecc:7s} {w:8.4f} ' + \
            f'{M:8.4f} {n:11.8f}{0:5d}'

    line2_checksum = tle_checksum(line2)
    line2 += str(line2_checksum)

    return line1, line2

class TLE():
    '''Two line telement

    Args:
        line1 (str): First line of Two-Line-Element set.
        line2 (str): Second line of Two-Line-Element set.

    Attributes:
        line1 (str): Human readable string describing the exception.
        line2 (str): Exception error code.
    '''

    def __init__(self, line1:str=None, line2:str=None, wgs:str='wgs84'):

        # Validate Input Lines
        self._validate_tle_input(line1, 1)
        self._validate_tle_input(line2, 2)

        # Set values for line
        self.line1 = line1
        self.line2 = line2

        # Set TLE Epoch
        epoch_year = int(line1[18:20])

        # Apply correction for TLEs only using 2-digit year
        if epoch_year < 57:
            epoch_year += 2000
        else:
            epoch_year += 1900
            
        doy = float(line1[20:32])
        self.epoch = Epoch(epoch_year, 1, 1, 0, 0, 0, tsys='UTC') + (doy-1)*86400.0

        # Create Internal SGP Propgator
        earth_model = None
        if wgs == 'wgs84':
            earth_model = sgp4.earth_gravity.wgs84
        elif wgs == 'wgs72':
            earth_model = sgp4.earth_gravity.wgs72
        else:
            raise RuntimeError(f'Unknown SGP Earth Gravity Model "{wgs:s}". Must be one of: wgs72,wgs84 (default).')

        self._sgp = sgp4.io.twoline2rv(self.line1, self.line2, earth_model)

    def _validate_tle_input(self, line:str, line_number:int):
        '''Internal validation method 

        Args:
            line (str): Line for validation
        '''

        line_checksum = tle_checksum(line)
        if line_checksum != int(line[-1]):
            raise RuntimeError(f'Invalid TLE checksum on line {line_number:1d}. Expected {line_checksum:1d}, found {line[-1]:s}.')

        if not validate_tle_line(line):
            raise RuntimeError(f'Invalid input TLE on line {line_number:1d}.')
        

    def _time_since_epoch(self, t:typing.Union[int, float,Epoch]) -> float:
        '''Compute elapsed time since TLE epoch in minutes since elapsed minutes
        is the standard input to the SGP4 propagator

        Args:
            t (:obj:`Epoch`): Time as either an Epoch or time since epoch in seconds.

        Returns:
            float: Elapsed time since epoch in minutes.
        '''

        if type(t) == float or type(t) == int:
            return t/60.0
        elif type(t) == Epoch:
            return (t - self.epoch)/60.0

    # TLE Properties
    @property
    def n(self):
        '''Mean motion of TLE object.

        Returns:
            float: Mean motion of TLE object. Units: [rev/day]
        '''

        return float(self.line2[52:63])

    @property
    def a(self):
        '''Semi-major axis of TLE object.

        Returns:
            float: Semi-major axis of TLE object. Units: [m]
        '''

        # Mean Motion [rev/day]
        n = self.n

        # Mean Motion [rad/s]
        n = n * 2.0 * math.pi / 86400.0

        return _astro.semimajor_axis(n)

    @property
    def e(self):
        '''Eccentricity of TLE object.

        Returns:
            float: Eccentricity of TLE object. Units: [rev/day]
        '''

        return float(f'0.{self.line2[26:33]}')

    @property
    def i(self):
        '''Inclination of TLE object.

        Returns:
            float: Inclination of TLE object. Units: [deg]
        '''

        return float(self.line2[8:16])

    @property
    def RAAN(self):
        '''Right ascension of ascending node of TLE object.

        Returns:
            float: Right ascension of ascending node of TLE object. Units: [deg]
        '''

        return float(self.line2[17:25])

    @property
    def w(self):
        '''Argument of Perigee of TLE object.

        Returns:
            float: Argument of Perigee of TLE object. Units: [deg]
        '''

        return float(self.line2[34:41])

    @property
    def M(self):
        '''Mean anomaly of TLE object.

        Returns:
            float: Mean anomaly of TLE object. Units: [deg]
        '''

        return float(self.line2[43:51])

    @property
    def ndt2(self):
        '''Second derivative of mean motion divided by 2 of TLE object.

        Returns:
            float: Second derivative of mean motion divided by 2 of TLE object. Units: [deg]
        '''

        return float(f'{self.line1[33]}0.{self.line1[35:43]}')

    @property
    def nddt6(self):
        '''Second derivative of mean motion divided by 2 of TLE object.

        Returns:
            float: Second derivative of mean motion divided by 2 of TLE object. Units: [deg]
        '''

        return float(f'{self.line1[44]}0.{self.line1[45:50]}e{self.line1[50:52]}')

    @property
    def bstar(self):
        '''B-star term of TLE object.

        Returns:
            float: B-star term of TLE object. Units: [deg]
        '''

        return float(f'{self.line1[53]}0.{self.line1[54:59]}e{self.line1[59:61]}')
    
    @property
    def tle_elements(self):
        '''Orbital elements comprising TLE
        '''

        return np.array([self.n, self.e, self.i, self.RAAN, self.w, self.M, self.ndt2, self.nddt6, self.bstar])

    @property
    def elements(self):
        '''Orbital elements comprising TLE
        '''

        return np.array([self.a, self.e, self.i, self.RAAN, self.w, self.M])
        
    # TLE State Propagation
    def state(self, t:typing.Union[float,Epoch]) -> np.ndarray:
        '''Return satellite state in default TLE output frame using the SGP4
        p

        Args:
            t (:obj:`Epoch`): Time as either an Epoch or time since epoch in seconds.

        Returns:
            np.ndarray: Return 
        '''

        # Get elapsed time in minutes
        dt = self._time_since_epoch(t)

        # Propagate state to time since epoch
        r, v, = sgp4.propagation.sgp4(self._sgp, dt)
        
        return np.hstack((r,v))*1.0e3

    def state_teme(self, t:typing.Union[float,Epoch]) -> np.ndarray:
        '''Compute the satellite state at the time in the inertial (TEME) frame.

        Args:
            t (:obj:`Epoch`): Time as either an Epoch or time since epoch in seconds.

        Returns:
            np.ndarray: Satellite state (position and velocity) at Epoch. Units: [m ; m/s]
        '''

        # Pass through call which is inertial
        return self.state(t)

    def state_pef(self, t:typing.Union[float,Epoch]) -> np.ndarray:
        '''Compute the satellite state at the time in the pseudo-Earth-fixed (PEF) frame.

        Args:
            t (:obj:`Epoch`): Time as either an Epoch or time since epoch in seconds.

        Returns:
            np.ndarray: Satellite state (position and velocity) at Epoch. Units: [m ; m/s]
        '''

        # Get state in ECI frame
        x_teme = self.state(t)

        # Extract components
        r_eci = x_teme[0:3]
        v_eci = x_teme[3:6]

        # Compute TEME -> ECEF transformation
        if type(t) == float or type(t) == int:
            epc = self.epoch + t
        else:
            epc = t
        

        R = _att.Rz(tle_gmst82(epc))
        omega_earth = np.array([0, 0, _constants.OMEGA_EARTH])

        # Compute ECEF state
        # Apply Polar Motion and Earth Rotation corrections.
        # Precession and Nutation Corrections are NOT applied since they are
        # already accounted for in the TEME frame
        r_pef = R @ r_eci
        v_pef = R @ v_eci - np.cross(omega_earth, R @ r_eci)

        return np.hstack((r_pef, v_pef))


    def state_itrf(self, t:typing.Union[float,Epoch]) -> np.ndarray:
        '''Compute the satellite state at the time in the ITRF Earth-Fixed (ECEF) frame.

        Args:
            t (:obj:`Epoch`): Time as either an Epoch or time since epoch in seconds.

        Returns:
            np.ndarray: Satellite state (position and velocity) at Epoch. Units: [m ; m/s]
        '''

        # Get state in ECI frame
        x_pef = self.state_pef(t)

        # Extract components
        r_pef = x_pef[0:3]
        v_pef = x_pef[3:6]

        # Compute TEME -> ECEF transformation
        if type(t) == float or type(t) == int:
            epc = self.epoch + t
        else:
            epc = t
        
        PM = _frames.polar_motion(epc)

        # Compute ECEF state
        # Apply Polar Motion and Earth Rotation corrections.
        # Precession and Nutation Corrections are NOT applied since they are
        # already accounted for in the TEME frame
        r_ecef = PM @ r_pef
        v_ecef = PM @ v_pef

        return np.hstack((r_ecef, v_ecef))

    def state_ecef(self, t:typing.Union[float,Epoch]) -> np.ndarray:
        '''Compute the satellite state at the time in the Earth-Fixed (ECEF) frame.
        The ECEF frame used here is the ITRF frame.

        Args:
            t (:obj:`Epoch`): Time as either an Epoch or time since epoch in seconds.

        Returns:
            np.ndarray: Satellite state (position and velocity) at Epoch. Units: [m ; m/s]
        '''
        return self.state_itrf(t)
        
    def state_gcrf(self, t:typing.Union[float,Epoch]) -> np.ndarray:
        '''Compute the satellite state at the time in the inertial (GCRF) frame.

        Args:
            t (:obj:`Epoch`): Time as either an Epoch or time since epoch in seconds.

        Returns:
            np.ndarray: Satellite state (position and velocity) at Epoch. Units: [m ; m/s]
        '''

        # Transform TEME -> ITRF
        x_itrf = self.state_itrf(t)

        if type(t) == float or type(t) == int:
            epc = self.epoch + t
        else:
            epc = t

        # Transform ITRF -> GCRF
        x_gcrf = _frames.sECEFtoECI(epc, x_itrf)

        # Return Transformation
        return x_gcrf

    def state_eci(self, t:typing.Union[float,Epoch]) -> np.ndarray:
        '''Compute the satellite state at the time in the inertial (GCRF) frame.

        Args:
            t (:obj:`Epoch`): Time as either an Epoch or time since epoch in seconds.

        Returns:
            np.ndarray: Satellite state (position and velocity) at Epoch. Units: [m ; m/s]
        '''

        # Pass through call which is inertial
        return self.state_gcrf(t)