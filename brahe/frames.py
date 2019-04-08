# -*- coding: utf-8 -*-
"""This module provides functions to convert between reference frames.

Note:
    Most transformations rely on the `pysofa2 <https://github.com/duncaneddy/pysofa2/>`_
    module to provide fast and accurate conversions. pysofa2 is a wrapper for the
    SOFA C implementation of the IAU reference frame conventions. The full
    source code can be found here: <http://www.iausofa.org/>
"""

# Module Imports
import numpy   as _np
import pysofa2 as _sofa
import brahe.constants as _constants
from   brahe.earthmodels import EOP as _EOP

#######################################
# IAU 2010 | Inertial <-> Earth-Fixed #
#######################################


def bias_precession_nutation(epc):
    """Computes the Bias-Precession-Nutation matrix transforming the GCRS to the 
    CIRS intermediate reference frame. This transformation corrects for the 
    bias, precession, and nutation of Celestial Intermediate Origin (CIO) with
    respect to inertial space.

    Args:
        epc (Epoch): Epoch of transformation

    Returns:
        rc2i (np.ndarray): 3x3 Rotation matrix transforming GCRS -> CIRS
    """

    # Constants of IAU 2006A transofrmation
    DMAS2R =  4.848136811095359935899141e-6 / 1.0e3
    dx06   =  0.0001750*DMAS2R
    dy06   = -0.0002259*DMAS2R

    # Compute X, Y, s terms using low-precision series terms
    x, y, s = _sofa.Xys00b(_constants.MJD_ZERO, epc.mjd(tsys="TT"))

    # Apply IAU2006 Offsets
    x += dx06
    y += dy06

    # Compute transformation and return
    rc2i = _sofa.C2ixys(x, y, s)

    return rc2i

def earth_rotation(epc):
    """Computes the Earth rotation matrix transforming the CIRS to the TIRS
    intermediate reference frame. This transformation corrects for the Earth
    rotation.

    Args:
        epc (Epoch): Epoch of transformation

    Returns:
        r (np.ndarray): 3x3 Rotation matrix transforming CIRS -> TIRS
    """

    # Compute Earth rotation angle
    era = _sofa.Era00(_constants.MJD_ZERO, epc.mjd(tsys="UT1"))

    # Rotate Matrix and return
    r = _sofa.Rz(era, _np.eye(3))

    return r

def polar_motion(epc):
    """Computes the Earth rotation matrix transforming the TIRS to the ITRF reference 
    frame.

    Args:
        epc (Epoch): Epoch of transformation

    Returns:
        rpm (np.ndarray): 3x3 Rotation matrix transforming TIRS -> ITRF
    """

    xp, yp = _EOP.pole_locator(epc.mjd(tsys="UTC"))

    # Compute transformation and return
    rpm = _sofa.Pom00(xp, yp, _sofa.Sp00(_constants.MJD_ZERO, epc.mjd(tsys="TT")))

    return rpm

def rECItoECEF(epc):
    """Computes the combined rotation matrix from the inertial to the Earth-fixed
    reference frame. Applies corrections for bias, precession, nutation,
    Earth-rotation, and polar motion.

    The transformation is accomplished using the IAU 2006/2000A, CIO-based 
    theory using classical angles. The method as described in section 5.5 of 
    the SOFA C transformation cookbook.

    Args:
        epc (Epoch): Epoch of transformation

    Returns:
        r (np.ndarray): 3x3 Rotation matrix transforming GCRF -> ITRF
    """

    # Compute intermediate transformations
    rc2i = bias_precession_nutation(epc)
    r    = earth_rotation(epc)
    rpm  = polar_motion(epc) 

    return rpm @ r @ rc2i

def rECEFtoECI(epc):
    """Computes the combined rotation matrix from the Earth-fixed to the inertial
    reference frame. Applies corrections for bias, precession, nutation,
    Earth-rotation, and polar motion.

    The transformation is accomplished using the IAU 2006/2000A, CIO-based 
    theory using classical angles. The method as described in section 5.5 of 
    the SOFA C transformation cookbook.

    Args:
        epc (Epoch): Epoch of transformation

    Returns:
        r (np.ndarray): 3x3 Rotation matrix transforming ITRF -> GCRF
    """

    # Compute intermediate transformations
    rc2i = bias_precession_nutation(epc)
    r    = earth_rotation(epc)
    rpm  = polar_motion(epc) 

    return rc2i.T @ r.T @ rpm.T

def sECItoECEF(epc, x):
    """Transforms an Earth inertial state into an Earth fixed state.

    The transformation is accomplished using the IAU 2006/2000A, CIO-based 
    theory using classical angles. The method as described in section 5.5 of 
    the SOFA C transformation cookbook.

    Args:
        epc (Epoch): Epoch of transformation
        x (np.ndarray): Inertial state (position, velocity). Units: [*m*; *m/s*]

    Returns:
        x_ecef (np.ndarray): Earth-fixed state (position, velocity)
    """

    # Ensure input is array-like
    x = _np.asarray(x)

    dim_x  = len(x)
    x_ecef = _np.zeros((dim_x,))

    # Extract State Components
    r_eci = x[0:3]

    if dim_x >= 6:
        v_eci = x[3:6]

    if dim_x == 9:
        a_eci = x[6:9]

    # Compute Sequential Transformation Matrices
    rc2i = bias_precession_nutation(epc)
    r    = earth_rotation(epc)
    pm   = polar_motion(epc)

    # Create Earth's Angular Rotation Vector
    omega_vec = _np.array([0, 0, _constants.OMEGA_EARTH]) # Neglect LOD effect

    # Calculate ECEF State
    x_ecef[0:3] = pm @  r @ rc2i @ r_eci

    if dim_x == 6:
        x_ecef[3:6] = pm @ (r @ rc2i @ v_eci - _np.cross(omega_vec, r @ rc2i @ r_eci))

    
    if dim_x == 9:
        x_ecef[6:9] = pm @ (r @ rc2i @ a_eci - _np.cross(omega_vec, _np.cross(omega_vec, r @ rc2i @ r_eci)) 
                                         - 2 * _np.cross(omega_vec, r @ rc2i @ v_eci))


    return x_ecef

def sECEFtoECI(epc, x):
    """Transforms an Earth fixed state into an Inertial state

    The transformation is accomplished using the IAU 2006/2000A, CIO-based 
    theory using classical angles. The method as described in section 5.5 of 
    the SOFA C transformation cookbook.

    Args:
        epc (Epoch): Epoch of transformation
        x (np.ndarray): Earth-fixed state (position, velocity) [*m*; *m/s*]

    Returns:
        x_ecef (np.ndarray): Inertial state (position, velocity)
    """

    # Ensure input is array-like
    x = _np.asarray(x)

    # Set state variable size
    dim_x = len(x)
    x_eci = _np.zeros((dim_x,))

    # Extract State Components
    r_ecef = x[0:3]

    if dim_x >= 6:
        v_ecef = x[3:6]


    if dim_x == 9:
        a_ecef = x[6:9]

    # Compute Sequential Transformation Matrices
    bpn = bias_precession_nutation(epc)
    rot = earth_rotation(epc)
    pm  = polar_motion(epc)

    # Create Earth's Angular Rotation Vector
    omega_vec = _np.array([0, 0, _constants.OMEGA_EARTH]) # Neglect LOD effect
    
    # Calculate ECEF State
    x_eci[0:3] = (pm @ rot @ bpn).T @ r_ecef

    if dim_x >= 6:
        x_eci[3:6] = (rot @ bpn).T @ (pm.T @ v_ecef + _np.cross(omega_vec, pm.T @ r_ecef))


    if dim_x >= 9:
        x_eci[6:9] = (rot @ bpn).T @ (pm.T @ a_ecef + _np.cross(omega_vec, _np.cross(omega_vec, pm.T @ x_eci[3:6])) 
                                 + 2 * _np.cross(omega_vec, pm.T @ x_eci[6:9])) 


    return x_eci