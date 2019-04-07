# -*- coding: utf-8 -*-
"""This data provides function to download or update the data files required to
use the brahe package.
"""

# Imports
import logging
import typing as _typing
import math   as _math
import copy   as _copy
import numpy  as _np

import brahe.constants as _constants

# Get Logger
logger = logging.getLogger(__name__)

####################
# Common Functions #
####################

def mean_motion(a:float, use_degrees:bool=False, gm:float=_constants.GM_EARTH):
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

def semimajor_axis(n:float, use_degrees:bool=False, gm:float=_constants.GM_EARTH):
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

def orbital_period(a:float, gm:float=_constants.GM_EARTH):
    """Compute the orbital period given the semi-major axis.

    Arguments:
        a (float): Semi-major axis. Units: *m*
        gm (float): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

    Returns:
        T (float): Orbital period. Units: *s*
    """
    
    return 2.0*_math.pi*_math.sqrt(a**3/gm)


def sun_sync_inclination(a:float, e:float, use_degrees:bool=False):
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

def anm_eccentric_to_mean(E:float, e:float, use_degrees:bool=False):
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

def anm_mean_to_eccentric(M:float, e:float, use_degrees:bool=False):
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
    while abs(f) > eps:
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

def sOSCtoCART(x_oe, use_degrees:bool=False, gm:float=_constants.GM_EARTH):
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
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

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
    P    = _np.zeros(3)
    P[0] = _math.cos(omega)*_math.cos(RAAN) - _math.sin(omega)*_math.cos(i)*_math.sin(RAAN)
    P[1] = _math.cos(omega)*_math.sin(RAAN) + _math.sin(omega)*_math.cos(i)*_math.cos(RAAN)
    P[2] = _math.sin(omega)*_math.sin(i)

    Q    = _np.zeros(3)
    Q[0] = -_math.sin(omega)*_math.cos(RAAN) - _math.cos(omega)*_math.cos(i)*_math.sin(RAAN)
    Q[1] = -_math.sin(omega)*_math.sin(RAAN) + _math.cos(omega)*_math.cos(i)*_math.cos(RAAN)
    Q[2] =  _math.cos(omega)*_math.sin(i)

    # Find 3-Dimensional Position
    x      = _np.zeros(6)
    x[0:3] = a*(_math.cos(E)-e)*P + a*_math.sqrt(1-e*e)*_math.sin(E)*Q
    x[3:6] = _math.sqrt(gm*a)/_np.linalg.norm(x[0:3])*(-_math.sin(E)*P + _math.sqrt(1-e*e)*_math.cos(E)*Q)

    return x

def sCARTtoOSC(x, use_degrees:bool=False, gm:float=_constants.GM_EARTH):
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
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

    Returns:
        x_oe (np.array_like): Osculating orbital elements. See above for 
            desription of the elements and their required order.
    """

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
    M     = anm_eccentric_to_mean(E, e)                     # Mean Anomaly
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