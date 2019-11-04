# -*- coding: utf-8 -*-
"""This module provides functions to convert between different representations
of orbital coordinates. As well as functions to compute properties of the orbits
themselves.
"""

# Imports
import logging as _logging
import typing as _typing
import math   as _math
import copy   as _copy
import numpy  as _np

# Brahe Imports
from brahe.utils import logger
from brahe.utils import AbstractArray
import brahe.constants as _constants

# Get Logger
logger = _logging.getLogger(__name__)

###########################
# Astrodynamic Properties #
###########################

def mean_motion(a:float, use_degrees:bool=False, gm:float=_constants.GM_EARTH) -> float:
    """Compute the mean motion given a semi-major axis.

    Args:
        a (float): Semi-major axis. Units: *m*
        use_degrees (bool): If ``True`` returns result in units of degrees.
        gm (float): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

    Returns:
        n (float): Orbital mean motion. Units: *rad/s* or *deg/s*
    """

    n = _math.sqrt(gm/a**3)

    if use_degrees:
        n *= 180.0/_math.pi

    return n

def semimajor_axis(n:float, use_degrees:bool=False, gm:float=_constants.GM_EARTH) -> float:
    """Compute the semi-major axis given the mean-motion

    Args:
        n (float): Orbital mean motion. Units: *rad/s* or *deg/s*
        use_degrees (bool): If ``True`` returns result in units of degrees.
        gm (float): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

    Returns:
        a (float): Semi-major axis. Units: *m*
    """

    if use_degrees:
        n *= _math.pi/180.0

    a = (gm/n**2)**(1.0/3.0)

    return a

def orbital_period(a:float, gm:float=_constants.GM_EARTH) -> float:
    """Compute the orbital period given the semi-major axis.

    Arguments:
        a (float): Semi-major axis. Units: *m*
        gm (float): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

    Returns:
        T (float): Orbital period. Units: *s*
    """
    
    return 2.0*_math.pi*_math.sqrt(a**3/gm)


def sun_sync_inclination(a:float, e:float, use_degrees:bool=False) -> float:
    """Compute the required inclination for a Sun-synchronous Earth orbit.

    Algorithm assumes the nodal precession is entirely due to the J2 perturbation, 
    and no other perturbations are considered. The inclination is computed using 
    a first-order, non-iterative approximation.

    Args:
        a (float): Semi-major axis. Units: *m*
        e (float) Eccentricity. Units: *dimensionless*
        use_degrees (bool): Return output in degrees (Default: false)

    Returns:
        i (float): Requierd inclination for a sun-synchronous orbit. Units:
            *rad* or *deg*
    """

    # Compute the required RAAN precession of a sun-synchronous orbit
    OMEGA_DOT_SS = 2.0*_math.pi/365.2421897/86400.0

    # Inclination required for sun-synchronous orbits
    i = _math.acos(-2.0 * a**(7/2) * OMEGA_DOT_SS * (1-e**2)**2 / 
            (3*(_constants.R_EARTH**2) * _constants.J2_EARTH * _math.sqrt(_constants.GM_EARTH)))

    if use_degrees:
        i *= 180.0/_math.pi

    return i

def anm_eccentric_to_mean(E:float, e:float, use_degrees:bool=False) -> float:
    """Convert eccentric anomaly into mean anomaly.

    Args:
        E (float): Eccentric anomaly. Units: *rad* or *deg*
        e (float): Eccentricity. Units: *dimensionless*
        use_degrees (bool): Handle input and output in degrees (Default: false)

    Returns:
        M (float): Mean anomaly. Units: *rad* or *deg*
    """

    # Convert degree input
    if use_degrees:
        E *= _math.pi/180.0

    # Convert eccentric to mean
    M = E - e*_math.sin(E)

    # Convert degree output
    if use_degrees:
        M *= 180.0/_math.pi

    return M

def anm_mean_to_eccentric(M:float, e:float, use_degrees:bool=False) -> float:
    """Convert mean anomaly into eccentric anomaly.

    Args:
        M (float): Mean anomaly. Units: *rad* or *deg*
        e (float): Eccentricity. Units: *dimensionless*
        use_degrees (bool): Handle input and output in degrees (Default: false)

    Returns:
        E (float): Eccentric anomaly. Units: *rad* or *deg*
    """

    # Convert degree input
    if use_degrees:
        M *= _math.pi/180.0

    # Convert mean to eccentric
    max_iter = 15
    eps      = _np.finfo(float).eps*100

    # Initialize starting values
    M = _math.fmod(M, 2.0*_math.pi)
    if e < 0.8:
        E = M
    else:
        E = _math.pi

    # Initialize working variable
    f = E - e*_math.sin(E) - M
    i = 0

    # Iterate until convergence
    while _math.fabs(f) > eps:
        f = E - e*_math.sin(E) - M
        E = E - f / (1.0 - e*_math.cos(E))

        # Increase iteration counter
        i += 1
        if i == max_iter:
            raise RuntimeError("Maximum number of iterations reached before convergence.")

    # Convert degree output
    if use_degrees:
        E *= 180.0/_math.pi

    return E

##########################
# Orbital Element States #
##########################

def sOSCtoCART(x_oe:AbstractArray, use_degrees:bool=False, gm:float=_constants.GM_EARTH) -> _np.ndarray:
    """Given an orbital state expressed in osculating orbital elements compute 
    the equivalent Cartesean position and velocity of the inertial state.

    The osculating elements are assumed to be (in order):

        1. _a_, Semi-major axis Units: *m*
        2. _e_, Eccentricity. Units: *dimensionless*
        3. _i_, Inclination. Units: *rad* or *deg*
        4. _Ω_, Right Ascension of the Ascending Node (RAAN). Units: *rad*
        5. _ω_, Argument of Perigee. Units: *rad* or *deg*
        6. _M_, Mean anomaly. Units: *rad* or *deg*

    Args:
        x_oe (np.array_like): Osculating orbital elements. See above for desription of
            the elements and their required order.
        use_degrees (bool): Handle input and output in degrees (Default: ``False``)
        gm (float): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m**3/s**2*

    Returns:
        x (np.array_like): Cartesean inertial state. Returns position and 
            velocity. Units: [*m*; *m/s*]
    """

    # Ensure input is numpy array
    x_oe = _np.asarray(x_oe)

    # Conver input 
    if use_degrees:
        # Copy and convert input from degrees to radians if necessary
        oe = _copy.deepcopy(x_oe)
        oe[2:6] = oe[2:6]*_math.pi/180.0
    else:
        oe = x_oe
    
    # Unpack input
    a, e, i, RAAN, omega, M = oe

    E = anm_mean_to_eccentric(M, e)

    # Create perifocal coordinate vectors
    P    = _np.zeros((3,))
    P[0] = _math.cos(omega)*_math.cos(RAAN) - _math.sin(omega)*_math.cos(i)*_math.sin(RAAN)
    P[1] = _math.cos(omega)*_math.sin(RAAN) + _math.sin(omega)*_math.cos(i)*_math.cos(RAAN)
    P[2] = _math.sin(omega)*_math.sin(i)

    Q    = _np.zeros((3,))
    Q[0] = -_math.sin(omega)*_math.cos(RAAN) - _math.cos(omega)*_math.cos(i)*_math.sin(RAAN)
    Q[1] = -_math.sin(omega)*_math.sin(RAAN) + _math.cos(omega)*_math.cos(i)*_math.cos(RAAN)
    Q[2] =  _math.cos(omega)*_math.sin(i)

    # Find 3-Dimensional Position
    x      = _np.zeros((6,))
    x[0:3] = a*(_math.cos(E)-e)*P + a*_math.sqrt(1-e*e)*_math.sin(E)*Q
    x[3:6] = _math.sqrt(gm*a)/_np.linalg.norm(x[0:3])*(-_math.sin(E)*P + _math.sqrt(1-e*e)*_math.cos(E)*Q)

    return x

def sCARTtoOSC(x:AbstractArray, use_degrees:bool=False, gm:float=_constants.GM_EARTH) -> _np.ndarray:
    """Given a Cartesean position and velocity in the inertial frame, return the 
    state expressed in terms of  osculating orbital elements.

    The osculating elements are assumed to be (in order):

        1. _a_, Semi-major axis Units: *m*
        2. _e_, Eccentricity. Units: *dimensionless*
        3. _i_, Inclination. Units: *rad* or *deg*
        4. _Ω_, Right Ascension of the Ascending Node (RAAN). Units: *rad*
        5. _ω_, Argument of Perigee. Units: *rad* or *deg*
        6. _M_, Mean anomaly. Units: *rad* or *deg*

    Args:
        x (np.array_like): Cartesean inertial state. Returns position and 
            velocity. Units: [*m*; *m/s*]
        use_degrees (bool): Handle input and output in degrees (Default: ``False``)
        gm (float): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m**3/s**2*

    Returns:
        x_oe (np.array_like): Osculating orbital elements. See above for 
            desription of the elements and their required order.
    """

    # Ensure input is numpy array
    x = _np.asarray(x)

    # Initialize Cartesian Polistion and Velocity
    r = x[0:3]
    v = x[3:6]

    h = _np.cross(r, v)  # Angular momentum vector
    W = h/_np.linalg.norm(h)    # Unit vector along angular momentum vector

    i     = _math.atan2(_math.sqrt(W[0]*W[0] + W[1]*W[1]), W[2])         # Compute inclination
    OMEGA = _math.atan2(W[0], -W[1])                                     # Right ascension of ascending node                     # Compute RAAN
    p     = _np.linalg.norm(h)*_np.linalg.norm(h)/gm                     # Semi-latus rectum
    a     = 1.0/(2.0/_np.linalg.norm(r) - _np.linalg.norm(v)*_np.linalg.norm(v)/gm)    # Semi-major axis
    n     = _math.sqrt(gm/(a**3))                                        # Mean motion

    # Numerical stability hack for circular and near-circular orbits
    # Ensures that (1-p/a) is always positive
    if _np.isclose(a, p, atol=1e-9, rtol=1e-8):
        p = a

    e     = _math.sqrt(1 - p/a)                             # Eccentricity
    E     = _math.atan2(_np.dot(r, v)/(n*a**2), (1-_np.linalg.norm(r)/a))    # Eccentric Anomaly
    M     = anm_eccentric_to_mean(E, e)              # Mean Anomaly
    u     = _math.atan2(r[2], -r[0]*W[1] + r[1]*W[0])       # Mean longiude
    nu    = _math.atan2(_math.sqrt(1-e*e)*_math.sin(E), _math.cos(E)-e)  # True Anomaly
    omega = u - nu                                          # Argument of perigee

    # Correct angles to run from 0 to 2PI
    OMEGA = OMEGA + 2.0*_math.pi
    omega = omega + 2.0*_math.pi
    M     = M     + 2.0*_math.pi

    OMEGA = _math.fmod(OMEGA, 2.0*_math.pi)
    omega = _math.fmod(omega, 2.0*_math.pi)
    M     = _math.fmod(M,     2.0*_math.pi)

    # Create Orbital Element Vector
    x_oe = _np.array([a, e, i, OMEGA, omega, M])

    # Convert output to degrees if necessary
    if use_degrees:
        x_oe[2:6] = x_oe[2:6]*180.0/_math.pi

    return x_oe

######################
# Earth-Fixed States #
######################

# Square of eccentricity
_ECC2 = _constants.WGS84_f * (2.0 - _constants.WGS84_f)

def sGEOCtoECEF(geoc:AbstractArray, use_degrees:bool=False) -> _np.ndarray:
    """Convert geocentric position to equivalent Earth-fixed position.

    Args:
        geoc (np.ndarray): Geocentric coordinates (lon, lat, altitude). Units: *rad* or *deg* and *m*
        use_degrees (bool): Handle input and output in degrees. (Default: ``False``)

    Returns:
        ecef (np.ndarray): Earth-fixed coordinates. Units *m*
    """

    # Extract lat and lon
    lon = geoc[0]
    lat = geoc[1]
    
    # Handle non-explict use-degrees
    if len(geoc) == 3:
        alt = geoc[2]
    else:
        alt = 0.0

    # Convert input to radians
    if use_degrees:
        lat = lat*_math.pi/180.0
        lon = lon*_math.pi/180.0

    # Check validity of input
    if lat < -_math.pi/2 or lat > _math.pi/2:
        raise RuntimeError(f"Lattiude, {(lat*180.0/_math.pi):.2f}, out of range. Must be between -90 and 90 degrees.")

    # Compute Earth fixed coordinates
    r       = _constants.WGS84_a + alt
    x = r*_math.cos(lat)*_math.cos(lon)
    y = r*_math.cos(lat)*_math.sin(lon)
    z = r*_math.sin(lat)
    
    return _np.array([x, y, z])

def sECEFtoGEOC(ecef:AbstractArray, use_degrees:bool=False) -> _np.ndarray:
    """Convert Earth-fixed position to geocentric location.

    Args:
        ecef (np.ndarray): Earth-fixed coordinates. Units *m*
        use_degrees (bool): Handle input and output in degrees. (Default: ``False``)

    Returns:
        geoc (np.ndarray): Geocentric coordinates (lon, lat, altitude). Units: *rad* or *deg* and *m*
    """
    
    # Expand ECEF coordinates
    x, y, z = ecef

    # Compute geocentric coordinates
    lat = _math.atan2(z, _math.sqrt(x*x + y*y))
    lon = _math.atan2(y, x)
    alt = _math.sqrt(x*x + y*y + z*z) - _constants.WGS84_a

    # Convert output to degrees
    if use_degrees:
        lat = lat*180.0/_math.pi
        lon = lon*180.0/_math.pi

    return _np.array([lon, lat, alt])

def sGEODtoECEF(geod:AbstractArray, use_degrees:bool=False) -> _np.ndarray:
    """Convert geodettic position to equivalent Earth-fixed position.

    Args:
        geod (np.ndarray): Geodetic coordinates (lon, lat, altitude). Units: *rad* or *deg* and *m*
        use_degrees (bool): Handle input and output in degrees. (Default: ``False``)

    Returns:
        ecef (np.ndarray): Earth-fixed coordinates. Units *m*
    """

    # Extract lat and lon
    lon = geod[0]
    lat = geod[1]
    
    # Handle non-explict use-degrees
    if len(geod) == 3:
        alt = geod[2]
    else:
        alt = 0.0

    # Convert input to radians
    if use_degrees:
        lat = lat*_math.pi/180.0
        lon = lon*_math.pi/180.0

    # Check validity of input
    if lat < -_math.pi/2 or lat > _math.pi/2:
        raise RuntimeError(f"Lattiude, {(lat*180.0/_math.pi):.2f}, out of range. Must be between -90 and 90 degrees.")

    # Compute Earth-fixed position vector
    N = _constants.WGS84_a / _math.sqrt(1.0 - _ECC2*_math.sin(lat)**2)
    x =           (N+alt)*_math.cos(lat)*_math.cos(lon)
    y =           (N+alt)*_math.cos(lat)*_math.sin(lon)
    z =  ((1.0-_ECC2)*N+alt)*_math.sin(lat)
    
    return _np.array([x, y, z])

def sECEFtoGEOD(ecef:AbstractArray, use_degrees:bool=False) -> _np.ndarray:
    """Convert Earth-fixed position to geodetic location.

    Args:
        ecef (np.ndarray): Earth-fixed coordinates. Units *m*
        use_degrees (bool): Handle input and output in degrees. (Default: ``False``)

    Returns:
        geod (np.ndarray): Geodettic coordinates (lon, lat, altitude). Units: *rad* or *deg* and *m*
    """

    # Expand ECEF coordinates
    x, y, z = ecef

    # Compute intermediate quantities
    epsilon  = _np.finfo(float).eps * 1.0e3 * _constants.WGS84_a # Convergence requirement as function of machine precision
    rho2     = x**2 + y**2                                         # Square of the distance from the z-axis
    dz       = _ECC2 * z
    N        = 0.0

    # Iteratively compute refine coordinates
    while True:
        zdz    = z + dz
        Nh     = _math.sqrt(rho2 + zdz**2)
        sinphi = zdz / Nh
        N      = _constants.WGS84_a / _math.sqrt(1.0 - _ECC2 * sinphi**2)
        dz_new = N * _ECC2 * sinphi

        # Check convergence requirement
        if _math.fabs(dz - dz_new) < epsilon:
            break

        dz = dz_new

    # Extract geodetic coordinates
    zdz = z + dz
    lat = _math.atan2(zdz, _math.sqrt(rho2))
    lon = _math.atan2(y, x)
    alt = _math.sqrt(rho2 + zdz**2) - N

    # Convert output to degrees
    if use_degrees:
        lat = lat*180.0/_math.pi
        lon = lon*180.0/_math.pi

    return _np.array([lon, lat, alt])

######################
# Topocentric States #
######################

def rECEFtoENZ(ecef:AbstractArray, conversion:str="geodetic") -> _np.ndarray:
    """Compute the rotation matrix from the Earth-fixed to the East-North-Zenith
    coorindate basis.

    Args:
        station_ecef (np.ndarray): Earth-fixed Cartesian station coordinates. Units: *m*
        conversion (bool): Conversion type to use. Can be "geocentric" or "geodetic"

    Returns:
        E (np.ndarray): Topocentric rotation matrix
    """
    
    if len(ecef) < 3:
        raise RuntimeError("Input coordinates must be length 3.")

    # Ensure input is numpy array
    ecef = _np.asarray(ecef)

    # Compute Station Lat-Lon-Altitude
    if conversion == "geodetic":
        lat, lon, _ = sECEFtoGEOD(ecef, use_degrees=False)
    elif conversion == "geocentric":
        lat, lon, _ = sECEFtoGEOC(ecef, use_degrees=False)
    else:
        raise RuntimeError(f"Unknown conversion method: {conversion}")


    # Compute ENZ basis vectors
    eE = _np.array([[-_math.sin(lon)], [_math.cos(lon)], [0]])
    eN = _np.array([[-_math.sin(lat)*_math.cos(lon)], [-_math.sin(lat)*_math.sin(lon)], [_math.cos(lat)]])
    eZ = _np.array([[_math.cos(lat)*_math.cos(lon)], [_math.cos(lat)*_math.sin(lon)], [_math.sin(lat)]])

    # Construct Rotation matrix
    E = _np.hstack((eE, eN, eZ)).T

    # Return Result
    return E

def rENZtoECEF(ecef:AbstractArray, conversion:str="geodetic") -> _np.ndarray:
    """Compute the rotation matrix from the the East-North-Zenith to the
    Earth-Centered-Earth-Fixed coorindate basis.

    Args:
        station_ecef (np.ndarray): Earth-fixed Cartesian station coordinates
        conversion (bool): Conversion type to use. Can be "geocentric" or "geodetic"

    Returns:
        E (np.ndarray): Topocentric rotation matrix
    """
    # Check input coordinates
    if len(ecef) < 3:
        raise RuntimeError("Input coordinates must be length 3.")

    return rECEFtoENZ(ecef, conversion=conversion).T

def sECEFtoENZ(station_ecef:AbstractArray, ecef:AbstractArray, conversion:str="geodetic") -> _np.ndarray:
    """Compute the coordinates of an object in East-Noth-Zenith topocentric
    coordinate basis of a given fixed-location station.

    Args:
        station_ecef (np.ndarray): Earth-fixed Cartesian station coordinates
        ecef (np.ndarray): Eartth-fixed coordinates of the object
        conversion (bool): Conversion type to use. Can be "geocentric" or "geodetic"

    Returns:
        x (np.ndarray): Object coordinates in East-North-Zenith basis.
    """

    # Check input sizes
    if len(ecef) < 3:
        raise RuntimeError("Input ecef state must be at least length 3.")

    if len(station_ecef) < 3:
        raise RuntimeError("Input station coordinates must be length 3.")

    # Ensure inputs are numpy arrays
    station_ecef = _np.asarray(station_ecef)
    ecef = _np.asarray(ecef)

    # Compute ENZ Rotation matrix
    E = rECEFtoENZ(station_ecef, conversion=conversion)

    # Transform range
    range_ecef = ecef[0:3] - station_ecef
    range_enz  = E @ range_ecef

    # Transform range-rate (if necessary)
    if len(ecef) == 6:
        range_rate_ecef = ecef[3:6]
        range_rate_enz  = E @ range_rate_ecef

    # Return
    if len(ecef) == 6:
        sat_enz = _np.hstack((range_enz, range_rate_enz))
    else:
        sat_enz = range_enz
    
    return sat_enz

def sENZtoECEF(station_ecef:AbstractArray, enz:AbstractArray, conversion:str="geodetic") -> _np.ndarray:
    """Compute the Earth-fixed coordinates of an object given East-Noth-Zenith
    topocentric coordinates, and the Earth-fixed coordinates of the station.

    Args:
        station_ecef (np.ndarray): Earth-fixed Cartesian station coordinates
        enz (np.ndarray): ENZ coordinates of the object
        conversion (bool): Conversion type to use. Can be "geocentric" or "geodetic"

    Returns:
        x (np.ndarray): Object coordinates in Earth-fixed basis.
    """
    
    # Check input sizes
    if len(enz) < 3:
        raise RuntimeError("Input ENZ state must be at least length 3.")

    if len(station_ecef) < 3:
        raise RuntimeError("Input station coordinates must be length 3.")

    # Ensure inputs are numpy arrays
    station_ecef = _np.asarray(station_ecef)
    enz = _np.asarray(enz)

    # Compute ENZ Rotation matrix
    E = rENZtoECEF(station_ecef, conversion=conversion)

    # Transform range
    range_enz  = enz[0:3]
    range_ecef = E @ range_enz

    # Transform range-rate (if necessary)
    if len(enz) == 6:
        range_rate_enz  = enz[3:6]
        range_rate_ecef = E @ range_rate_enz

    # Return
    if len(enz) == 6:
        sat_ecef = _np.hstack((range_ecef + station_ecef, range_rate_ecef))
    else:
        sat_ecef = range_ecef + station_ecef
    
    return sat_ecef

def rECEFtoSEZ(ecef:AbstractArray, conversion:str="geodetic") -> _np.ndarray:
    """Compute the rotation matrix from the Earth-fixed to the South-East-Zenith 
    coorindate basis.

    Args:
        station_ecef (np.ndarray): Earth-fixed Cartesian station coordinates
        conversion (bool): Conversion type to use. Can be "geocentric" or "geodetic"

    Returns:
        E (np.ndarray): Topocentric rotation matrix
    """

    if len(ecef) < 3:
        raise RuntimeError("Input coordinates must be length 3.")

    # Compute Station Lat-Lon-Altitude
    if conversion == "geodetic":
        lat, lon, _ = sECEFtoGEOD(ecef, use_degrees=False)
    elif conversion == "geocentric":
        lat, lon, _ = sECEFtoGEOC(ecef, use_degrees=False)
    else:
        raise RuntimeError(f"Unknown conversion method: {conversion}")

    # Ensure inputs are numpy arrays
    ecef = _np.asarray(ecef)

    # Compute SEZ basis vectors
    eS = _np.array([[_math.sin(lat)*_math.cos(lon)], [_math.sin(lat)*_math.sin(lon)], [-_math.cos(lat)]])
    eE = _np.array([[-_math.sin(lon)], [_math.cos(lon)], [0]])
    eZ = _np.array([[_math.cos(lat)*_math.cos(lon)], [_math.cos(lat)*_math.sin(lon)], [_math.sin(lat)]])

    # Construct Rotation matrix
    E = _np.hstack((eS, eE, eZ)).T

    # Return Result
    return E


def rSEZtoECEF(ecef:AbstractArray, conversion:str="geodetic") -> _np.ndarray:
    """Compute the rotation matrix from the Earth-fixed to the South-East-Zenith 
    coorindate basis.

    Args:
        station_ecef (np.ndarray): Earth-fixed Cartesian station coordinates
        conversion (bool): Conversion type to use. Can be "geocentric" or "geodetic"

    Returns:
        E (np.ndarray): Topocentric rotation matrix
    """

    # Check input coordinates
    if len(ecef) < 3:
        raise RuntimeError("Input coordinates must be length 3.")

    # Ensure inputs are numpy arrays
    ecef = _np.asarray(ecef)

    return rECEFtoSEZ(ecef, conversion=conversion).T



def sECEFtoSEZ(station_ecef:AbstractArray, ecef:AbstractArray, conversion:str="geodetic") -> _np.ndarray:
    """Compute the coordinates of an object in South-East-Zenith topocentric
    coordinate basis of a given fixed-location station.

    Args:
        station_ecef (np.ndarray): Earth-fixed Cartesian station coordinates
        ecef (np.ndarray): Eartth-fixed coordinates of the object
        conversion (bool): Conversion type to use. Can be "geocentric" or "geodetic"

    Returns:
        x (np.ndarray): SEZ coordinates of object
    """

    # Check input sizes
    if len(ecef) < 3:
        raise RuntimeError("Input ecef state must be at least length 3.")

    if len(station_ecef) < 3:
        raise RuntimeError("Input station coordinates must be length 3.")

    # Ensure inputs are numpy arrays
    station_ecef = _np.asarray(station_ecef)
    ecef         = _np.asarray(ecef)

    # Construct SEZ Rotation matrix
    E = rECEFtoSEZ(station_ecef, conversion=conversion)

    # Transform range
    range_ecef = ecef[0:3] - station_ecef
    range_sez  = E @ range_ecef

    # Transform range-rate (if necessary)
    if len(ecef) == 6:
        range_rate_ecef = ecef[3:6]
        range_rate_sez  = E @ range_rate_ecef

    # Return
    if len(ecef) == 6:
        sez = _np.hstack((range_sez, range_rate_sez))
    else:
        sez = range_sez
    
    return sez

def sSEZtoECEF(station_ecef:AbstractArray, sez:AbstractArray, conversion:str="geodetic") -> _np.ndarray:
    """Compute the coordinates of an object in the topocentric frame of an
    Earth-fixed frame

    Args:
        station_ecef (np.ndarray): Earth-fixed Cartesian station coordinates
        sez (np.ndarray): SEZ coordinates of the object
        conversion (bool): Conversion type to use. Can be "geocentric" or "geodetic"

    Returns:
        x (np.ndarray): Earth-fixed coordinates of the object
    """

    # Check input sizes
    if len(sez) < 3:
        raise RuntimeError("Input SEZ state must be at least length 3.")

    if len(station_ecef) < 3:
        raise RuntimeError("Input station coordinates must be length 3.")

    # Ensure inputs are numpy arrays
    station_ecef = _np.asarray(station_ecef)
    sez          = _np.asarray(sez)

    # Compute ENZ Rotation matrix
    E = rSEZtoECEF(station_ecef, conversion=conversion)

    # Transform range
    range_sez  = sez[0:3]
    range_ecef = E @ range_sez

    # Transform range-rate (if necessary)
    if len(sez) >= 6:
        range_rate_sez  = sez[3:6]
        range_rate_ecef = E @ range_rate_sez

    # Return
    if len(sez) >= 6:
        sat_ecef = _np.hstack((range_ecef + station_ecef, range_rate_ecef))
    else:
        sat_ecef = range_ecef + station_ecef
    
    return sat_ecef

def sENZtoAZEL(x:AbstractArray, use_degrees:bool=False) -> _np.ndarray:
    """Convert East-North-Zenith topocentric coordinates into
    azimuth, elevation, and range. Azimuth-rate, elevation-rate, and range-rate
    will also be computed is the object velocity if the ENZ-coordinates are also
    provided.

    Args:
        x (np.ndarray): East-North-Up coordinates.
        use_degrees (bool): If ``True`` returns result in units of degrees.

    Returns:
        azel (np.ndarray): Azimuth, elevation and range. Units: *rad* or *deg* and *m*
    """

    # Check inputs
    if not (len(x) == 3 or len(x) == 6):
        raise RuntimeError("Input ENZ state must be length 3 or 6.")

    # Expand values
    rE, rN, rZ = x[0:3]
    
    # Range
    rho = _np.linalg.norm(x[0:3])

    # Elevation
    el = _math.atan2(rZ, _math.sqrt(rE**2 + rN**2))

    # Azimuth
    az = 0.0
    if el != _math.pi/2:
        # Non-singular azimuth 
        az = _math.atan2(rE, rN)
        if az < 0:
            az += 2*_math.pi
    else:
        # Azimuth may be singular for 90 deg elevation
        if len(x) != 6:
            az = 0.0
        else:
            # Use rate information to get azimuth if there is a singularity
            # in the position
            az = _math.atan2(x[3], x[4])

    # Output
    azel = _np.array([az, el, rho])

    if use_degrees:
        azel[0] *= 180.0/_math.pi
        azel[1] *= 180.0/_math.pi

    # Process Rate information
    if len(x) == 6:
        rdE, rdN, rdZ = x[3:6]

        # Range-rate
        rhod = _np.dot(x[0:3], x[3:6])/rho

        # Elevation-rate
        eld = (rdZ - _np.linalg.norm(x[3:6])*_math.sin(el))/_math.sqrt(rE**2 + rN**2)

        # Azimuth-rate
        azd = (rdE*rN - rdN*rE)/(rE**2 + rN**2)

        # Output
        azel_rate = _np.array([azd, eld, rhod])

        if use_degrees:
            azel_rate[0] *= 180/_math.pi
            azel_rate[1] *= 180/_math.pi

    # Return
    if len(x) == 6:
        return _np.hstack((azel, azel_rate))
    else:
        return azel

def sSEZtoAZEL(x:AbstractArray, use_degrees:bool=False) -> _np.ndarray:
    """Convert South-East-Zenith topocentric coordinates into
    azimuth, elevation, and range. Azimuth-rate, elevation-rate, and range-rate
    will also be computed is the object velocity if the SEZ-coordinates are also
    provided.

    Args:
        x (np.ndarray): South-East-Zenith coordinates.
        use_degrees (bool): If `True` returns result in units of degrees.

    Returns:
        azel (np.ndarray): Azimuth, elevation and range [rad; rad; m]
    """

    # Check inputs
    if not (len(x) == 3 or len(x) == 6):
        raise RuntimeError("Input rECEFtoSEZ state must be length 3 or 6.")

    # Expand values
    rS, rE, rZ = x[0:3]
    
    # Range
    rho = _np.linalg.norm(x[0:3])

    # Elevation
    el = _math.atan2(rZ, _math.sqrt(rS**2 + rE**2))

    # Azimuth
    az = 0.0
    if el != _math.pi/2:
        az = _math.atan2(rE, -rS)
        if az < 0:
            az += 2*_math.pi
    else:
        if len(x) != 6:
            az = 0.0
        else:
            # Use rate information to get azimuth if there is a singularity
            # in the position
            az = _math.atan2(x[5], -x[4])

    # Output
    azel = _np.array([az, el, rho])

    if use_degrees:
        azel[0] *= 180.0/_math.pi
        azel[1] *= 180.0/_math.pi

    # Process Rate information
    if len(x) == 6:
        rdS, rdE, rdZ = x[3:6]

        # Range-rate
        rhod = _np.dot(x[0:3], x[3:6])/rho

        # Elevation-rate
        eld = (rdZ - _np.linalg.norm(x[3:6])*_math.sin(el))/_math.sqrt(rS**2 + rE**2)

        # Azimuth-rate
        azd = (rdS*rE - rdE*rS)/(rS**2 + rE**2)

        # Output
        azel_rate = _np.array([azd, eld, rhod])
        if use_degrees:
            azel_rate[0] *= 180/_math.pi
            azel_rate[1] *= 180/_math.pi

    # Return
    if len(x) == 6:
        return _np.hstack((azel, azel_rate))
    else:
        return azel