"""This orbit dynamics submoduble provides functions for computing acceleration 
due to special relativity
"""

import numba
import numpy as np

import brahe.constants as _const

@numba.jit(nopython=True, cache=True)
def accel_relativity(x:np.ndarray) -> np.ndarray:
    '''Computes perturbation accleration of a satellite in the Inertial frame
    due to the combined effects of special and general relativity.

    Args:
        x (:obj:`np.ndarray`): Satellite Cartesean state in the inertial reference frame [m; m/s]

    Returns:
        np.ndarray: Satellite acceleration due to relativity. [m/s^2]

    References:
        1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.110-112.
    '''
    # Extract state variables
    r = x[0:3]
    v = x[3:6]

    # Intermediate computations
    norm_r = np.linalg.norm(r)
    r2     = norm_r**2

    norm_v = np.linalg.norm(v)
    v2     = norm_v**2

    c  = _const.C_LIGHT
    c2 = c**2

    # Compute unit vectors
    er = r/norm_r
    ev = v/norm_v

    # Compute perturbation and return
    a_rel = _const.GM_EARTH/r2 * ( (4*_const.GM_EARTH/(c2*norm_r) - v2/c2)*er + 4*v2/c2*np.dot(er, ev)*ev)

    return a_rel