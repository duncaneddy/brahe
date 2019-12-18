'''Provide fundamental geometry calculations used by the scheduling.
'''

import math
import numpy as np

import brahe.data_models as bdm
from brahe.constants import RAD2DEG
from brahe.coordinates import sECEFtoENZ, sENZtoAZEL, sECEFtoGEOD, sGEODtoECEF


def azelrng(sat_ecef: np.ndarray,
            loc_ecef: np.ndarray,
            use_degrees: bool = True) -> np.ndarray:
    '''Compute satellite azimuth, elevation, and range as viewed from the specified location.

    Args:
        sat_ecef (:obj:`np.ndarray`): Satellite position in the ECEF frame
        loc_ecef (:obj:`np.ndarray`): Location in ECEF (ITRF) frame.
        use_degrees (:obj:`bool`, optional): Return output in degrees. Default: `True`

    Returns:
        np.ndarray: azimuth elevation and range as array [deg, deg, m]
    '''

    # Ensure np-ness
    sat_ecef = np.asarray(sat_ecef)
    loc_ecef = np.asarray(loc_ecef)

    # Compute Satellite State in ENZ frame
    sat_enz = sECEFtoENZ(loc_ecef[0:3], sat_ecef[0:3], conversion='geodetic')

    # Compute Satellite Elevation
    azelrng = sENZtoAZEL(sat_enz, use_degrees=use_degrees)[0:3]

    return azelrng


def azimuth(sat_ecef: np.ndarray,
            loc_ecef: np.ndarray,
            use_degrees: bool = True) -> np.ndarray:
    '''Compute satellite azimuth as viewed from the specified location.

    Args:
        sat_ecef (:obj:`np.ndarray`): Satellite position in the ECEF frame
        loc_ecef (:obj:`np.ndarray`): Location in ECEF (ITRF) frame.
        use_degrees (:obj:`bool`, optional): Return output in degrees. Default: `True`

    Returns:
        float: Azimuth [deg]
    '''

    return azelrng(sat_ecef, loc_ecef, use_degrees=use_degrees)[0]


def elevation(sat_ecef: np.ndarray,
              loc_ecef: np.ndarray,
              use_degrees: bool = True) -> np.ndarray:
    '''Compute satellite elevation as viewed from the specified location.

    Args:
        sat_ecef (:obj:`np.ndarray`): Satellite position in the ECEF frame
        loc_ecef (:obj:`np.ndarray`): Location in ECEF (ITRF) frame.
        use_degrees (:obj:`bool`, optional): Return output in degrees. Default: `True`

    Returns:
        float: Elevation [deg]
    '''

    return azelrng(sat_ecef, loc_ecef, use_degrees=use_degrees)[1]


def range(sat_ecef: np.ndarray, loc_ecef: np.ndarray,
          use_degrees: bool = True) -> np.ndarray:
    '''Compute satellite range from the specified location.

    Args:
        sat_ecef (:obj:`np.ndarray`): Satellite position in the ECEF frame.
        loc_ecef (:obj:`np.ndarray`): Location in ECEF (ITRF) frame.
        use_degrees (:obj:`bool`, optional): Return output in degrees. Default: `True`

    Returns:
        float: Range [m]
    '''

    return azelrng(sat_ecef, loc_ecef, use_degrees=use_degrees)[2]


def look_angle(sat_ecef: np.ndarray,
                    loc_ecef: np.ndarray,
                    use_degrees: bool = True) -> np.ndarray:
    '''Compute the look angle angle between the satellite and the specific location.

    Args:
        sat_ecef (:obj:`np.ndarray`): Satellite position in the ECEF frame.
        loc_ecef (:obj:`np.ndarray`): Location in ECEF (ITRF) frame.
        use_degrees (:obj:`bool`, optional): Return output in degrees. Default: `True`

    Returns:
        float: look angle angle [deg]
    '''

    # Ensure np-ness
    sat_ecef = np.asarray(sat_ecef)
    loc_ecef = np.asarray(loc_ecef)

    # Satellite state
    r_sat = sat_ecef[0:3]

    # Geodetic sub-satellte point
    sat_geod = sECEFtoGEOD(r_sat)
    sub_sat_geod = np.array([sat_geod[0], sat_geod[1], 0.0])
    sub_sat_ecef = sGEODtoECEF(sub_sat_geod)

    # look angle
    nadir_dir = (sub_sat_ecef - r_sat) / np.linalg.norm(sub_sat_ecef - r_sat)
    target_dir = (loc_ecef - r_sat) / np.linalg.norm(loc_ecef - r_sat)
    look_angle = math.acos(np.dot(nadir_dir, target_dir)) * RAD2DEG

    return look_angle


def ascdsc(sat_ecef: np.ndarray) -> bdm.AscendingDescending:
    '''Compute whether whether satellite is ascending or descending in current
    state.

    Args:
        sat_ecef (:obj:`np.ndarray`): Satellite position in the ECEF frame.
        use_degrees (:obj:`bool`, optional): Return output in degrees. Default: `True`

    Returns:
        bdm.AscendingDescending: ascending or descending state 
    '''

    # Ensure np-ness
    sat_ecef = np.asarray(sat_ecef)

    if sat_ecef[5] > 0:
        return bdm.AscendingDescending.ascending
    elif sat_ecef[5] < 0:
        return bdm.AscendingDescending.descending
    else:
        # Handle unlikely case that satellite is exaclty at 0 Z-velocity
        if sat_ecef[2] < 0:
            return bdm.AscendingDescending.ascending
        else:
            return bdm.AscendingDescending.descending


def look_direction(sat_ecef: np.ndarray,
                   loc_ecef: np.ndarray) -> bdm.LookDirection:
    '''Compute the look direction for viewing the startet

    Args:
        sat_ecef (:obj:`np.ndarray`): Satellite position in the ECEF frame.
        loc_ecef (:obj:`np.ndarray`): Location in ECEF (ITRF) frame.\

    Returns:
        bdm.LookDirection: Look direction. 'left' or 'right'
    '''

    # Ensure np-ness
    sat_ecef = np.asarray(sat_ecef)
    loc_ecef = np.asarray(loc_ecef)

    # Line of Sight Vector in ECEF Frame
    los_ecef = loc_ecef[0:3] - sat_ecef[0:3]

    # Transform from ECEF to TRN
    # TODO: Implement this in ATLAS
    # Ensure input is array-like

    # Compute RTN to ECEF rotation
    r = sat_ecef[0:3]
    v = sat_ecef[3:6]

    R = r / np.linalg.norm(r)
    N = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
    T = np.cross(N, R)

    R_rtn2ecef = np.hstack((R.reshape(-1, 1), T.reshape(-1,
                                                        1), N.reshape(-1, 1)))

    # Create RTN rotation matrix
    R_ecef2rtn = R_rtn2ecef.T

    los_rtn = R_ecef2rtn @ los_ecef

    # Compute cross product of RTN velocity and RTN LOS
    cp = np.cross([0, 1, 0], los_rtn)

    if np.sign(cp[0]) < 0:
        return bdm.LookDirection.right
    else:
        return bdm.LookDirection.left