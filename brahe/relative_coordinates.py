# -*- coding: utf-8 -*-
"""This module provides functions to convert between different representations
of relative orbital coordinates.
"""

# Imports
import logging as _logging
import typing  as _typing
import math    as _math
import numpy   as _np

# Brahe Imports
from   brahe.utils import logger
from brahe.utils import AbstractArray

def rRTNtoECI(x:AbstractArray) -> _np.ndarray:
    """Compute the radial, along-track, cross-track (RTN) rotation matrix. Which,
    if applied to a position vector in the RTN frame, will transform that vector to
    beinto the equivalent relative position in the ECI frame.

    The RTN frame is also commonly refered to as the local-vertical, local-horizontal (LVLH) frame.

    Arguments:
        x (np.ndarray): Inertial state (position and velocity) of primary 
            (observing) satellite. Units: [*m*; *m/s*]
        xt (np.ndarray): Inertial state (position and velocity) of the target 
            satellite. Units: [*m*; *m/s*]

    Returns:
        np.ndarray Rotation matrix transforming from
            the RTN frame to the ECI frame.
    """

    # Ensure input is array-like
    x = _np.asarray(x)

    r = x[0:3]
    v = x[3:6]

    R = r/_np.linalg.norm(r)
    N = _np.cross(r, v)/_np.linalg.norm(_np.cross(r, v))
    T = _np.cross(N, R)

    R_rtn2eci = _np.hstack((R.reshape(-1, 1), T.reshape(-1, 1), N.reshape(-1, 1)))

    return R_rtn2eci

def rECItoRTN(x:AbstractArray) -> _np.ndarray:
    """Compute the Earth-centered inertial to radial, along-track, cross-track (RTN) 
    rotation matrix. Which, if applied to a position vector in the ECI frame, will 
    transform that vector into the equivalent position vector in the RTN frame.

    The RTN frame is also commonly refered to as the local-vertical, local-horizontal (LVLH) frame.

    Arguments:
        x (np.ndarray): Inertial state (position and velocity) of primary 
            (observing) satellite. Units: [*m*; *m/s*]
        xt (np.ndarray): Inertial state (position and velocity) of the target 
            satellite.  Units: [*m*; *m/s*]

    Returns:
        np.ndarray: Rotation matrix transforming from the ECI 
        frame to the RTN frame.
    """
    
    # Ensure input is array-like
    x = _np.asarray(x)

    return rRTNtoECI(x).T

def sECItoRTN(x:AbstractArray, xt:AbstractArray) -> _np.ndarray:
    """Compute the radial, along-track, cross-track (RTN) coordinates of a target 
    satellite in the primary satellite's RTN frame.

    The RTN frame is also commonly refered to as the local-vertical,
    local-horizontal (LVLH) frame.

    Arguments:
        x (np.ndarray): Inertial state (position and velocity) of primary 
            (observing) satellite. Units: [*m*; *m/s*]
        xt (np.ndarray): Inertial state (position and velocity) of the target 
            satellite. Units: [*m*; *m/s*]

    Returns:
        np.ndarray: Position and velocity of the target relative of the 
            observing satellite in the RTN. Units: [*m*; *m/s*]
    """

    # Ensure input is array-like
    x  = _np.asarray(x)
    xt = _np.asarray(xt)

    # Create RTN rotation matrix
    R_eci2rtn = rECItoRTN(x)

    # Initialize output vector
    x_rtn = _np.zeros((6,)) if len(xt) >= 6 else _np.zeros((3,))

    # Transform Position
    r          = x[0:3]
    rho        = xt[0:3] - r

    x_rtn[0:3] = R_eci2rtn @ rho

    # Transform velocity
    if len(xt) >= 6:
        v          = x[3:6]
        f_dot      = _np.linalg.norm(_np.cross(r[:], v[:]))/_np.linalg.norm(r)**2
        omega      = _np.array([0.0, 0.0, f_dot])
        rho_dot    = xt[3:6] - v
        x_rtn[3:6] = R_eci2rtn @ rho_dot - _np.cross(omega, x_rtn[0:3])

    return x_rtn

def sRTNtoECI(x:AbstractArray, xrtn:AbstractArray) -> _np.ndarray:
    """Compute the Earth-center Inerttial coordinates of a satellite given the
    radial, along-track, cross-track coorinates in the observing satellite's
    RNT frame.

    The RTN frame is also commonly refered to as the local-vertical, 
    local-horizontal (LVLH) frame.

    Arguments:
        x (np.ndarray): Inertial state (position and velocity) of primary 
            (observing) satellite. Units: [*m*; *m/s*]
        rtn (np.ndarray): Position and velocity of the target relative of the 
            observing satellite in the RTN. Units: [*m*; *m/s*]

    Returns:
        np.ndarray: Inertial state (position and velocity) of the target 
            satellite. Units: [*m*; *m/s*]
    """

    # Ensure input is array-like
    x   = _np.asarray(x)
    rtn = _np.asarray(xrtn)

    # Create RTN rotation matrix
    R_rtn2eci = rRTNtoECI(x)

    # Initialize output vector
    xt = _np.zeros((6,)) if len(xrtn) >= 6 else _np.zeros((3,))

    # Transform position
    r       = x[0:3]
    r_rtn   = xrtn[0:3]
    xt[0:3] = R_rtn2eci @ r_rtn + r

    # Transform velocity
    if len(xrtn) >= 6:
        v = x[3:6]
        v_rtn   = xrtn[3:6]
        f_dot   = _np.linalg.norm(_np.cross(r, v))/_np.linalg.norm(r)**2
        omega   = _np.array([0.0, 0.0, f_dot])
        xt[3:6] = R_rtn2eci @ (v_rtn + _np.cross(omega, r_rtn)) + v

    return xt