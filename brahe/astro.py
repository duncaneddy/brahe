# -*- coding: utf-8 -*-
"""This module provides functions to convert between different representations
of orbital coordinates. As well as functions to compute properties of the orbits
themselves.
"""

# Imports
import logging as _logging
import typing as _typing
import math   as math
import copy   as copy
import numpy  as np

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
        a (:obj:`float`): Semi-major axis. Units: *m*
        use_degrees (bool): If ``True`` returns result in units of degrees.
        gm (:obj:`float`): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

    Returns:
        n (:obj:`float`): Orbital mean motion. Units: *rad/s* or *deg/s*
    """

    n = math.sqrt(gm/a**3)

    if use_degrees:
        n *= 180.0/math.pi

    return n

def semimajor_axis(n:float, use_degrees:bool=False, gm:float=_constants.GM_EARTH) -> float:
    """Compute the semi-major axis given the mean-motion

    Args:
        n (:obj:`float`): Orbital mean motion. Units: *rad/s* or *deg/s*
        use_degrees (bool): If ``True`` returns result in units of degrees.
        gm (:obj:`float`): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

    Returns:
        a (:obj:`float`): Semi-major axis. Units: *m*
    """

    if use_degrees:
        n *= math.pi/180.0

    a = (gm/n**2)**(1.0/3.0)

    return a

def orbital_period(a:float, gm:float=_constants.GM_EARTH) -> float:
    """Compute the orbital period given the semi-major axis.

    Arguments:
        a (:obj:`float`): Semi-major axis. Units: *m*
        gm (:obj:`float`): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

    Returns:
        T (:obj:`float`): Orbital period. Units: *s*
    """
    
    return 2.0*math.pi*math.sqrt(a**3/gm)


def perigee_velocity(a:_typing.Union[float, AbstractArray], e:float, gm:float=_constants.GM_EARTH) -> float:
    '''Compute the perigee velocity from orbital element state. Accepts input as
    semi-major axis and eccentricity or a Keplerian state vector.

    Args:
        a (:obj:`Union[float, AbstractArray]`): Input semi-major axis or Keplerian state vector
        e (:obj:`Union[float, AbstractArray]`): Input eccentricity. Unused if a vector is provided for first argument.
        gm (:obj:`float`): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

    Returns:
        v_per (:obj:`float`): Velocity at perigee
    '''

    # Expand array inputs
    if type(a) in [list, tuple, np.ndarray]:
        a, e = a[0], a[1]

    return math.sqrt(gm/a)*math.sqrt((1+e)/(1-e))

def apogee_velocity(a:_typing.Union[float, AbstractArray], e:float, gm:float=_constants.GM_EARTH) -> float:
    '''Compute the apogee velocity from orbital element state. Accepts input as
    semi-major axis and eccentricity or a Keplerian state vector.

    Args:
        a (:obj:`Union[float, AbstractArray]`): Input semi-major axis or Keplerian state vector
        e (:obj:`Union[float, AbstractArray]`): Input eccentricity. Unused if a vector is provided for first argument.
        gm (:obj:`float`): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m^3/s^2*

    Returns:
        v_apo (:obj:`float`): Velocity at apogee
    '''

    # Expand array inputs
    if type(a) in [list, tuple, np.ndarray]:
        a, e = a[0], a[1]

    return math.sqrt(gm/a)*math.sqrt((1-e)/(1+e))

def sun_sync_inclination(a:float, e:float, use_degrees:bool=False) -> float:
    """Compute the required inclination for a Sun-synchronous Earth orbit.

    Algorithm assumes the nodal precession is entirely due to the J2 perturbation, 
    and no other perturbations are considered. The inclination is computed using 
    a first-order, non-iterative approximation.

    Args:
        a (:obj:`float`): Semi-major axis. Units: *m*
        e (:obj:`float`) Eccentricity. Units: *dimensionless*
        use_degrees (obj:`bool`): Return output in degrees (Default: false)

    Returns:
        i (:obj:`float`): Required inclination for a sun-synchronous orbit. Units: *rad* or *deg*
    """

    # Compute the required RAAN precession of a sun-synchronous orbit
    OMEGA_DOT_SS = 2.0*math.pi/365.2421897/86400.0

    # Inclination required for sun-synchronous orbits
    i = math.acos(-2.0 * a**(7/2) * OMEGA_DOT_SS * (1-e**2)**2 / 
            (3*(_constants.R_EARTH**2) * _constants.J2_EARTH * math.sqrt(_constants.GM_EARTH)))

    if use_degrees:
        i *= 180.0/math.pi

    return i

def anm_eccentric_to_mean(E:float, e:float, use_degrees:bool=False) -> float:
    """Convert eccentric anomaly into mean anomaly.

    Args:
        E (:obj:`float`): Eccentric anomaly. Units: *rad* or *deg*
        e (:obj:`float`): Eccentricity. Units: *dimensionless*
        use_degrees (bool): Handle input and output in degrees (Default: false)

    Returns:
        M (:obj:`float`): Mean anomaly. Units: *rad* or *deg*
    """

    # Convert degree input
    if use_degrees:
        E *= math.pi/180.0

    # Convert eccentric to mean
    M = E - e*math.sin(E)

    # Convert degree output
    if use_degrees:
        M *= 180.0/math.pi

    return M

def anm_mean_to_eccentric(M:float, e:float, use_degrees:bool=False) -> float:
    """Convert mean anomaly into eccentric anomaly.

    Args:
        M (:obj:`float`): Mean anomaly. Units: *rad* or *deg*
        e (:obj:`float`): Eccentricity. Units: *dimensionless*
        use_degrees (bool): Handle input and output in degrees (Default: false)

    Returns:
        E (:obj:`float`): Eccentric anomaly. Units: *rad* or *deg*
    """

    # Convert degree input
    if use_degrees:
        M *= math.pi/180.0

    # Convert mean to eccentric
    max_iter = 15
    eps      = np.finfo(float).eps*100

    # Initialize starting values
    M = math.fmod(M, 2.0*math.pi)
    if e < 0.8:
        E = M
    else:
        E = math.pi

    # Initialize working variable
    f = E - e*math.sin(E) - M
    i = 0

    # Iterate until convergence
    while math.fabs(f) > eps:
        f = E - e*math.sin(E) - M
        E = E - f / (1.0 - e*math.cos(E))

        # Increase iteration counter
        i += 1
        if i == max_iter:
            raise RuntimeError("Maximum number of iterations reached before convergence.")

    # Convert degree output
    if use_degrees:
        E *= 180.0/math.pi

    return E

##########################
# Orbital Element States #
##########################

def sOSCtoCART(x_oe:AbstractArray, use_degrees:bool=False, gm:float=_constants.GM_EARTH) -> np.ndarray:
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
        gm (:obj:`float`): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m**3/s**2*

    Returns:
        x (np.array_like): Cartesean inertial state. Returns position and 
            velocity. Units: [*m*; *m/s*]
    """

    # Ensure input is numpy array
    x_oe = np.asarray(x_oe)

    # Conver input 
    if use_degrees:
        # Copy and convert input from degrees to radians if necessary
        oe = copy.deepcopy(x_oe)
        oe[2:6] = oe[2:6]*math.pi/180.0
    else:
        oe = x_oe
    
    # Unpack input
    a, e, i, RAAN, omega, M = oe

    E = anm_mean_to_eccentric(M, e)

    # Create perifocal coordinate vectors
    P    = np.zeros((3,))
    P[0] = math.cos(omega)*math.cos(RAAN) - math.sin(omega)*math.cos(i)*math.sin(RAAN)
    P[1] = math.cos(omega)*math.sin(RAAN) + math.sin(omega)*math.cos(i)*math.cos(RAAN)
    P[2] = math.sin(omega)*math.sin(i)

    Q    = np.zeros((3,))
    Q[0] = -math.sin(omega)*math.cos(RAAN) - math.cos(omega)*math.cos(i)*math.sin(RAAN)
    Q[1] = -math.sin(omega)*math.sin(RAAN) + math.cos(omega)*math.cos(i)*math.cos(RAAN)
    Q[2] =  math.cos(omega)*math.sin(i)

    # Find 3-Dimensional Position
    x      = np.zeros((6,))
    x[0:3] = a*(math.cos(E)-e)*P + a*math.sqrt(1-e*e)*math.sin(E)*Q
    x[3:6] = math.sqrt(gm*a)/np.linalg.norm(x[0:3])*(-math.sin(E)*P + math.sqrt(1-e*e)*math.cos(E)*Q)

    return x

def sCARTtoOSC(x:AbstractArray, use_degrees:bool=False, gm:float=_constants.GM_EARTH) -> np.ndarray:
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
        gm (:obj:`float`): Gravitational constant of central body. Defaults to 
            ``brahe.constants.GM_EARTH`` if not provided. Units: *m**3/s**2*

    Returns:
        x_oe (np.array_like): Osculating orbital elements. See above for 
            desription of the elements and their required order.
    """

    # Ensure input is numpy array
    x = np.asarray(x)

    # Initialize Cartesian Polistion and Velocity
    r = x[0:3]
    v = x[3:6]

    h = np.cross(r, v)  # Angular momentum vector
    W = h/np.linalg.norm(h)    # Unit vector along angular momentum vector

    i     = math.atan2(math.sqrt(W[0]*W[0] + W[1]*W[1]), W[2])         # Compute inclination
    OMEGA = math.atan2(W[0], -W[1])                                     # Right ascension of ascending node                     # Compute RAAN
    p     = np.linalg.norm(h)*np.linalg.norm(h)/gm                     # Semi-latus rectum
    a     = 1.0/(2.0/np.linalg.norm(r) - np.linalg.norm(v)*np.linalg.norm(v)/gm)    # Semi-major axis
    n     = math.sqrt(gm/(a**3))                                        # Mean motion

    # Numerical stability hack for circular and near-circular orbits
    # Ensures that (1-p/a) is always positive
    if np.isclose(a, p, atol=1e-9, rtol=1e-8):
        p = a

    e     = math.sqrt(1 - p/a)                             # Eccentricity
    E     = math.atan2(np.dot(r, v)/(n*a**2), (1-np.linalg.norm(r)/a))    # Eccentric Anomaly
    M     = anm_eccentric_to_mean(E, e)              # Mean Anomaly
    u     = math.atan2(r[2], -r[0]*W[1] + r[1]*W[0])       # Mean longiude
    nu    = math.atan2(math.sqrt(1-e*e)*math.sin(E), math.cos(E)-e)  # True Anomaly
    omega = u - nu                                          # Argument of perigee

    # Correct angles to run from 0 to 2PI
    OMEGA = OMEGA + 2.0*math.pi
    omega = omega + 2.0*math.pi
    M     = M     + 2.0*math.pi

    OMEGA = math.fmod(OMEGA, 2.0*math.pi)
    omega = math.fmod(omega, 2.0*math.pi)
    M     = math.fmod(M,     2.0*math.pi)

    # Create Orbital Element Vector
    x_oe = np.array([a, e, i, OMEGA, omega, M])

    # Convert output to degrees if necessary
    if use_degrees:
        x_oe[2:6] = x_oe[2:6]*180.0/math.pi

    return x_oe