# -*- coding: utf-8 -*-
"""This module provides access to planetary ephemerices in a manner consistent
with the rest of the brahe module.
"""

# Imports
import logging
import copy  as copy
import math  as math
import numpy as np

# Brahe Imports
from   brahe.utils import logger
import brahe.constants as _constants
from   brahe.attitude  import Rx
from   brahe.epoch     import Epoch

# Helper function
def _frac(x: float):
    return x-math.floor(x)

########################
# Analytic Ephemerides #
########################

def sun_position(epc:Epoch) -> np.ndarray:
    """Compute the Sun's position in the EME2000 inertial frame through the use
    of low-precision analytical functions.

    Args:
        epc: Epoch of ephemeris

    Returns:
        r_sun (np.ndarray): Position vector of the Sun in the Earth-centered 
            inertial frame.

    Notes:
    
        1. The EME2000 inertial frame is for most purposes equivalent to the GCRF 
            frame.

    References:
    
        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
            Applications*, 2012, p.70-73.
    """

    # Constants
    mjd_tt  = epc.mjd(tsys="TT")                    # MJD of epoch in TT
    epsilon = 23.43929111*math.pi/180.0            # Obliquity of J2000 ecliptic
    T       = (mjd_tt-_constants.MJD2000)/36525.0   # Julian cent. since J2000

    # Variables

    # Mean anomaly, ecliptic longitude and radius
    M = 2.0*math.pi * _frac(0.9931267 + 99.9973583*T)            # [rad]
    L = 2.0*math.pi * _frac(0.7859444 + M/(2.0*math.pi) + \
        (6892.0*math.sin(M)+72.0*math.sin(2.0*M)) / 1296.0e3)   # [rad]
    r = 149.619e9 - 2.499e9*math.cos(M) - 0.021e9*math.cos(2*M) # [m]

    # Equatorial position vector
    p_sun = Rx(-epsilon) @ np.array([r*math.cos(L), r*math.sin(L), 0.0])

    return p_sun

def moon_position(epc:Epoch) -> np.ndarray:
    """Compute the Moon's position in the EME2000 inertial frame through the use
    of low-precision analytical functions.

    Args:
        epc (Epoch) Epoch

    Returns:
        r_moon (np.ndarray): Position vector of the Moon in the Earth-centered 
            inertial ffame.

    Notes:
    
        1. The EME2000 inertial frame is for most purposes equivalent to the GCRF 
            frame.

    References:
    
        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
            Applications*, 2012, p.70-73.
    """

    # Constants
    mjd_tt  = epc.mjd(tsys="TT")                    # MJD of epoch in TT
    epsilon = 23.43929111*math.pi/180.0            # Obliquity of J2000 ecliptic
    T       = (mjd_tt-_constants.MJD2000)/36525.0   # Julian cent. since J2000

    # Mean elements of lunar orbit
    L_0 =     _frac(0.606433 + 1336.851344*T)          # Mean longitude [rev] w.r.t. J2000 equinox
    l   = 2.0*math.pi*_frac(0.374897 + 1325.552410*T) # Moon's mean anomaly [rad]
    lp  = 2.0*math.pi*_frac(0.993133 +   99.997361*T) # Sun's mean anomaly [rad]
    D   = 2.0*math.pi*_frac(0.827361 + 1236.853086*T) # Diff. long. Moon-Sun [rad]
    F   = 2.0*math.pi*_frac(0.259086 + 1342.227825*T) # Argument of latitude 

    # Ecliptic longitude (w.r.t. equinox of J2000)
    dL = + 22640*math.sin(l) - 4586*math.sin(l-2*D) + 2370*math.sin(2*D) +  769*math.sin(2*l) \
         - 668*math.sin(lp) - 412*math.sin(2*F) - 212*math.sin(2*l-2*D) - 206*math.sin(l+lp-2*D) \
         + 192*math.sin(l+2*D) - 165*math.sin(lp-2*D) - 125*math.sin(D) - 110*math.sin(l+lp) \
         + 148*math.sin(l-lp) - 55*math.sin(2*F-2*D)

    L = 2.0*math.pi * _frac(L_0 + dL/1296.0e3) # [rad]

    # Ecliptic latitude
    S  = F + (dL+412*math.sin(2*F)+541*math.sin(lp)) * _constants.AS2RAD 
    h  = F-2*D
    N  = - 526*math.sin(h) + 44*math.sin(l+h) - 31*math.sin(-l+h) - 23*math.sin(lp+h) \
         + 11*math.sin(-lp+h) - 25*math.sin(-2*l+F) + 21*math.sin(-l+F)
    B  = (18520.0*math.sin(S) + N) * _constants.AS2RAD   # [rad]

    # Distance [m]
    r = + 385000e3 - 20905e3*math.cos(l) - 3699e3*math.cos(2*D-l) - 2956e3*math.cos(2*D) \
        - 570e3*math.cos(2*l) + 246e3*math.cos(2*l-2*D) - 205e3*math.cos(lp-2*D) \
        - 171e3*math.cos(l+2*D) - 152e3*math.cos(l+lp-2*D)   

    # Equatorial coordinates
    p_moon = Rx(-epsilon) @ np.array([r*math.cos(L)*math.cos(B), r*math.sin(L)*math.cos(B), r*math.sin(B)])

    return p_moon