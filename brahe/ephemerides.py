# -*- coding: utf-8 -*-
"""This module provides access to planetary ephemerices in a manner consistent
with the rest of the brahe module.
"""

# Imports
import logging
import copy  as _copy
import math  as _math
import numpy as _np

import brahe.constants as _constants
from   brahe.attitude  import Rx
from   brahe.epoch     import Epoch

# Get Logger
logger = logging.getLogger(__name__)

# Helper function
def _frac(x: float):
    return x-_math.floor(x)

########################
# Analytic Ephemerides #
########################

def sun_position(epc:Epoch):
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
    epsilon = 23.43929111*_math.pi/180.0            # Obliquity of J2000 ecliptic
    T       = (mjd_tt-_constants.MJD2000)/36525.0   # Julian cent. since J2000

    # Variables

    # Mean anomaly, ecliptic longitude and radius
    M = 2.0*_math.pi * _frac(0.9931267 + 99.9973583*T)            # [rad]
    L = 2.0*_math.pi * _frac(0.7859444 + M/(2.0*_math.pi) + \
        (6892.0*_math.sin(M)+72.0*_math.sin(2.0*M)) / 1296.0e3)   # [rad]
    r = 149.619e9 - 2.499e9*_math.cos(M) - 0.021e9*_math.cos(2*M) # [m]

    # Equatorial position vector
    p_sun = Rx(-epsilon) @ _np.array([r*_math.cos(L), r*_math.sin(L), 0.0])

    return p_sun

def moon_position(epc:Epoch):
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
    epsilon = 23.43929111*_math.pi/180.0            # Obliquity of J2000 ecliptic
    T       = (mjd_tt-_constants.MJD2000)/36525.0   # Julian cent. since J2000

    # Mean elements of lunar orbit
    L_0 =     _frac(0.606433 + 1336.851344*T)          # Mean longitude [rev] w.r.t. J2000 equinox
    l   = 2.0*_math.pi*_frac(0.374897 + 1325.552410*T) # Moon's mean anomaly [rad]
    lp  = 2.0*_math.pi*_frac(0.993133 +   99.997361*T) # Sun's mean anomaly [rad]
    D   = 2.0*_math.pi*_frac(0.827361 + 1236.853086*T) # Diff. long. Moon-Sun [rad]
    F   = 2.0*_math.pi*_frac(0.259086 + 1342.227825*T) # Argument of latitude 

    # Ecliptic longitude (w.r.t. equinox of J2000)
    dL = + 22640*_math.sin(l) - 4586*_math.sin(l-2*D) + 2370*_math.sin(2*D) +  769*_math.sin(2*l) \
         - 668*_math.sin(lp) - 412*_math.sin(2*F) - 212*_math.sin(2*l-2*D) - 206*_math.sin(l+lp-2*D) \
         + 192*_math.sin(l+2*D) - 165*_math.sin(lp-2*D) - 125*_math.sin(D) - 110*_math.sin(l+lp) \
         + 148*_math.sin(l-lp) - 55*_math.sin(2*F-2*D)

    L = 2.0*_math.pi * _frac(L_0 + dL/1296.0e3) # [rad]

    # Ecliptic latitude
    S  = F + (dL+412*_math.sin(2*F)+541*_math.sin(lp)) * _constants.AS2RAD 
    h  = F-2*D
    N  = - 526*_math.sin(h) + 44*_math.sin(l+h) - 31*_math.sin(-l+h) - 23*_math.sin(lp+h) \
         + 11*_math.sin(-lp+h) - 25*_math.sin(-2*l+F) + 21*_math.sin(-l+F)
    B  = (18520.0*_math.sin(S) + N) * _constants.AS2RAD   # [rad]

    # Distance [m]
    r = + 385000e3 - 20905e3*_math.cos(l) - 3699e3*_math.cos(2*D-l) - 2956e3*_math.cos(2*D) \
        - 570e3*_math.cos(2*l) + 246e3*_math.cos(2*l-2*D) - 205e3*_math.cos(lp-2*D) \
        - 171e3*_math.cos(l+2*D) - 152e3*_math.cos(l+lp-2*D)   

    # Equatorial coordinates
    p_moon = Rx(-epsilon) @ _np.array([r*_math.cos(L)*_math.cos(B), r*_math.sin(L)*_math.cos(B), r*_math.sin(B)])

    return p_moon