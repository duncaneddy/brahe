'''This orbit dynamics submoduble provides functions for computing acceleration 
due to solar radiation pressure.
'''

import math
import numba
import numpy as np

import brahe.constants as _const

@numba.jit(nopython=True, cache=True)
def accel_srp(x:np.ndarray, r_sun:np.ndarray, mass:float=0, area:float=0, CR:float=1.8, p0:float=_const.P_SUN, au:float=_const.AU) -> np.ndarray:
    '''Computes the perturbing acceleration due to direct solar radiation 
    pressure assuming the reflecting surface is a flat plate pointed directly at
    the Sun.

    Args:
        x (:obj:`np.ndarray`): Satellite Cartesean state in the inertial reference frame [m; m/s]

    Returns:
        np.ndarray: Satellite acceleration due to srp. [m/s**2]

    References:
        1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.77-79.
    '''
    
    # Spacecraft position vector
    r = x[0:3]

    # Relative position vector of spacecraft w.r.t. Sun
    d = r - r_sun

    # Acceleration due to moon point mass
    a_srp = d * (CR*(area/mass)*p0*au**2 / np.linalg.norm(d)**3)

    # Return
    return a_srp

@numba.jit(nopython=True, cache=True)
def eclipse_cylindrical(x:np.ndarray, r_sun:np.ndarray) -> np.ndarray:
    '''Computes the illumination fraction of a satellite in Earth orbit using a
    cylindrical Earth shadow model.

    Arguments:
        x (:obj:`np.ndarray`): Satellite Cartesean state in the inertial reference frame [m; m/s]
        r_sun (:obj:`np.ndarray`): Position of sun in inertial frame.

    Returns:
        float: Illumination fraction (0 <= nu <= 1). nu = 0 means spacecraft in complete shadow, nu = 1 mean spacecraft fully illuminated by sun.

    References:
        1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.80-83.
    '''
    # Satellite inertial position
    r = x[0:3]
    
    # Sun-direction unit-vector
    e_sun = r_sun / np.linalg.norm(r_sun)
    
    # Projection of spacecraft position
    s = np.dot(r, e_sun)

    # Compute illumination faction
    nu = 0.0
    if s >= 1.0 or np.linalg.norm(r - s*e_sun) > _const.R_EARTH:
        nu = 1.0

    return nu

def eclipse_conical(x:np.ndarray, r_sun:np.ndarray) -> np.ndarray:
    '''Computes the illumination fraction of a satellite in Earth orbit using a
    conical Earth shadow model.

    Arguments:
        x (:obj:`np.ndarray`): Satellite Cartesean state in the inertial reference frame [m; m/s]
        r_sun (:obj:`np.ndarray`): Position of sun in inertial frame.

    Returns:
        float: Illumination fraction (0 <= nu <= 1). nu = 0 means spacecraft in complete shadow, nu = 1 mean spacecraft fully illuminated by sun.

    References:
        1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.80-83.
    '''

    # Satellite inertial position
    r = x[0:3]

    # Occultation Geometry
    a = math.asin(_const.R_SUN/np.linalg.norm(r_sun - r))
    b = math.asin(_const.R_EARTH/np.linalg.norm(r))
    c = math.acos(np.dot(r, r_sun-r)/(np.linalg.norm(r)*np.linalg.norm(r_sun-r)) )

    e_sun = r_sun / np.linalg.norm(r_sun)

    # Test Occulation Conditions
    nu = 0.0
    if math.fabs(a - b) < c and c < (a + b):
        # Partial occultation
    
        xx = (c**2 + a**2 - b**2)/(2*c)
        yy = math.sqrt(a**2 - xx**2)
        A  = a**2 * math.acos(xx/a) + b**2 * math.acos((c-xx)/b) - c * yy

        nu = 1 - A/(math.pi*a**2)
    elif (a + b) <= c:
        # No occultation
        nu = 1.0
    else:
        # Full occultation
        nu = 0.0
    
    return nu
