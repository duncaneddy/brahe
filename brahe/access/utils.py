'''Utility functions for tessellation.

In particular provides spherical geometry.
'''

import copy
import math
import numpy as np

import spherical_geometry.polygon as sgp
import spherical_geometry.great_circle_arc as sggca
import spherical_geometry.vector as sgv

from brahe.utils import fcross
from brahe.constants import R_EARTH, GM_EARTH, DEG2RAD, RAD2DEG
from brahe.epoch import Epoch
from brahe.coordinates import sECEFtoGEOD, sGEODtoECEF
from brahe.attitude import Rz
from brahe.astro import orbital_period, anm_eccentric_to_mean
from brahe.tle import TLE

import brahe.data_models as bdm

from . import access_geometry as ageo


def sphere_point_to_latlon(vec: np.ndarray):
    '''Convert a spherical geometry point to a vector. Accounts for wrapping of
    lognitude. Uses geodetic conversion to perform conversion.

    Args:
        vec (:obj:`np.ndarray`): Input point on unit sphere to lat lon.

    Returns:
        np.ndarray: Returns longitude, latitude, altitude. Units [deg; deg; m]
    '''

    vec = np.asarray(vec)

    ll = np.asarray(sECEFtoGEOD(vec * R_EARTH, use_degrees=True)[0:2])

    if ll[0] > 180.0:
        ll[0] -= 360.0

    return ll


def rodrigues_rotation(vector: np.ndarray, axis: np.ndarray, angle: float):
    '''Rotate a vector about an axis by angle by applying Rodrigues' rotation
    formula.

    Args:
        vector (:obj:`np.ndarray`): Vector to rotate.
        axis (:obj:`np.ndarray`): Axis of rotation.
        angle (float): Angle to rotate vector through about axis. Rotation
            occurs in counter-clockwise [deg].

    Returns:
        np.ndarray: Rotated vector.
    '''

    # Ensure np-ness
    vector = np.asarray(vector)
    axis = np.asarray(axis)

    return vector * math.cos(angle) + np.array(fcross(axis, vector)) * math.sin(angle) + axis * np.dot(axis, vector) * (1 - math.cos(angle))


def create_spherical_polygon(geojson: bdm.GeoJSONObject):
    '''Create a normalized spherical geometry polygon for use in tessellation computation.

    Args:
        geojson (:obj:`GeoJSONObject`): GeoJSON Polygon object.

    Returns:
        SphericalPolygon: Working data-type represtation of geojson object on unit sphere. Used by `spherical_geometry` package.
    '''

    if geojson.geotype != 'Polygon':
        raise RuntimeError(f'Input GeoJSON object time must be of "Polygon", not "{geojson.geotype}"')

    # Use Lon-Lat spherical polygon constructor because, I couldn't get the cartesian one to work
    lon, lat, _ = zip(*geojson.coordinates)
    return sgp.SphericalPolygon.from_radec(lon, lat)


def circumscription_length(geojson: bdm.GeoJSONObject):
    '''Compute minimum length radius of a circle that circumscribes the input
    polygon. The raidus is for acircle from the center point.

    Args:
        geojson (:obj:`GeoJSONObject`): GeoJSON Polygon to compute subscripting
            length for.

    Returns:
        float: Length of arc that circumscribes area [m]
    '''

    max_length = 0.0

    center = geojson.center
    cnt = sgv.radec_to_vector(center[0], center[1])

    # Iterate over all combinations of points
    for pnt in geojson.vertices:
        # Compute Spherical Geometry points
        v = sgv.radec_to_vector(pnt[0], pnt[1])

        # Compute length and scale by length of Earth
        ang = math.acos(np.dot(cnt, v))
        length = ang * R_EARTH

        if length > max_length:
            max_length = length

    return max_length


def compute_area(geojson: bdm.GeoJSONObject):
    '''Compute area of input GeoJSON polygon.

    Args:
        geojson (:obj:`GeoJSONObject`): GeoJSON Polygon

    Returns:
        float: Area of polygon [m^2]

    Assumptions:
        - Spherical Earth with radius equal to the equatorial radius
    '''

    # Create spherical polygon
    sp = create_spherical_polygon(geojson)

    # Calculate area in steradians
    return sp.area() * R_EARTH**2


def analytical_half_orbit(tle: TLE):
    '''Compute times of next ascending and descending half orbit using
    analysitc method based on determining the mean argument of latitude.

    Args:
        tle (:obj:`TLE`): TLE of object for tessellation.

    Returns:
        (Epoch, Epoch): Start and stop time of next ascending half orbit centered on TLE Epoch.
        (Epoch, Epoch): Start and stop time of next descending half orbit centered on TLE Epoch.
    '''

    # Set time if none provided
    epc = tle.epoch

    # Get Orbit Mean Motion and Semi-Major Axis
    elements = tle.elements

    n = tle.n * (2 * math.pi / 86400)
    a = (GM_EARTH / n**2)**(1.0 / 3.0)
    T = orbital_period(a)

    # Compute Argument of latitude
    e = elements[1]
    w = elements[4]
    M = elements[5]

    # Convert Target - 90 argument of latitude to mean anomaly
    Et = math.atan(math.sqrt(1 - e**2) * math.sin(90.0 * DEG2RAD) / (math.cos(90.0 * DEG2RAD) + e))
    Mt = anm_eccentric_to_mean(Et, e) * RAD2DEG

    # Get Descending Orbit
    dt = (Mt - w - M) * DEG2RAD / n

    epc_dsc_start = epc + dt
    epc_dsc_end = epc + dt + T / 2.0
    epc_asc_start = copy.deepcopy(epc_dsc_end)
    epc_asc_end = epc_asc_start + T / 2.0

    return (epc_asc_start, epc_asc_end), (epc_dsc_start, epc_dsc_end)

def find_latitude_crossing(tle: TLE, lat: float, epc_start: Epoch, epc_end: Epoch, 
                            timestep: float = 300, tol: float = 0.001):
    '''Find time of latitude crossing within given window.

    Args:
        orbit (:obj:`TLE`): Orbit object
        lat (float): Latitude to compute crossing for.
        epc_start (:obj:`Epoch`): Start of time window to search for crossing
        epc_end (:obj:`Epoch`): End of time window to search for crossing
        timestep (float, Default: 300): Initial timestep of search for latitude crossing.
        tol (float: Default: 0.001): Accuracy tolerance for time of latitude crossing.

    Returns:
        epc: Time of latitude crossing
        np.ndarray: ECEF state of satellite at crossing
    '''

    ## Helper function
    # Compute difference between satellite state and desired latitude
    def lat_error(tle: TLE, lat: float, epc: Epoch):
        sat_lat = sECEFtoGEOD(tle.state_itrf(epc)[0:3], use_degrees=True)[1]
        lat_err = lat - sat_lat

        return lat_err

    # Copy step epoch
    epc = copy.deepcopy(epc_start)

    # Determine desired direction based on velocity at mid time
    ascdsc_dir = ageo.ascdsc(tle.state_itrf(epc_start + (epc_end - epc_start) / 2.0))

    if ascdsc_dir == bdm.AscendingDescending.ascending:
        ascdir = 1
    else:
        ascdir = -1

    # Compute initial error
    lat_err = lat_error(tle, lat, epc)

    while math.fabs(lat_err) > tol:
        # Set step direction to lower error
        timestep = ascdir * np.sign(lat_err) * math.fabs(timestep)

        # Step
        epc += timestep

        # Calculate new error
        lat_err = lat_error(tle, lat, epc)

        # If sign doesn't match half size
        if ascdir * np.sign(lat_err) != np.sign(timestep):
            timestep = timestep / 2.0

        if epc < epc_start or epc > epc_end:
            raise RuntimeError(f'No latitude crossing found in window.')

    return epc, tle.state_itrf(epc)

def compute_along_track_directions(tle: TLE, point: np.ndarray):
    '''Compute the orbit along-track direction for when 

    Args:
        tle (:obj:`TLE`): Orbit object 
        point (:obj:`np.ndarray`): Geodetic coordinates of point

    Returns:
        Tuple(np.ndarray, np.ndarray): Sets of along-track crossings. 
    '''

    # Get Target latitude
    lon, lat = point[0], point[1]

    # Get Orbital Inclination to check if a crossing exits
    incl = tle.elements[2]

    if (math.fabs(lat) > incl and incl <= 90.0) or (math.fabs(lat) > (180 - incl) and incl > 90.0):
        raise RuntimeError(f'Desired latitude {lat:.2f} exceeds maximum latitude accesible by TLE {incl:.2f}')

    # Get half-orbits
    t_asc, t_dsc = analytical_half_orbit(tle)

    # Get ascending crossing
    _, asc_ecef = find_latitude_crossing(tle, lat, *t_asc)

    # Extract along-track direction
    asc_at = asc_ecef[3:6] / np.linalg.norm(asc_ecef[3:6])

    # Remove component normal to surface
    asc_at = asc_at - np.dot(asc_ecef[0:3] / np.linalg.norm(asc_ecef[0:3]), asc_at)
    asc_at = asc_at / np.linalg.norm(asc_at)

    # Rotate direction to be oriented locally at collect point
    sat_lon = sECEFtoGEOD(asc_ecef[0:3], use_degrees=True)[0]

    asc_at = Rz(-lon, use_degrees=True) @ Rz(sat_lon, use_degrees=True) @ asc_at

    # Get descending crossing
    _, dsc_ecef = find_latitude_crossing(tle, lat, *t_dsc)

    # Extract along-track direction
    dsc_at = dsc_ecef[3:6] / np.linalg.norm(dsc_ecef[3:6])

    # Remove component normal to surface
    dsc_at = dsc_at - np.dot(dsc_ecef[0:3] / np.linalg.norm(dsc_ecef[0:3]), dsc_at)
    dsc_at = dsc_at / np.linalg.norm(dsc_at)

    # Rotate direction to be oriented locally at collect point
    sat_lon = sECEFtoGEOD(dsc_ecef[0:3], use_degrees=True)[0]

    dsc_at = Rz(-lon, use_degrees=True) @ Rz(sat_lon, use_degrees=True) @ dsc_at

    return asc_at, dsc_at


def compute_crosstrack_width(geojson: bdm.GeoJSONObject, direction: np.ndarray):
    '''Compute cross-track width of image.

    Args:
        geojson (:obj:`GeoJSONObject`): Geojson object. Must be of type `Polygon`
        direction (:obj:`np.ndarray`): Direction for calculating cross track direction.

    Returns:
        float: Length in cross-track direction
        int: Index of minimal cross-track point
        float: Minimum cross-track distance
        int: Index of maximaul cross-track point
        float: Maximum cross-track distance
    '''

    if geojson.geotype != 'Polygon':
        raise RuntimeError(f'Function not defined for GeoJSON type {geojson.geotype}')

    # Compute Great Circle Arc
    cpnt = geojson.center_ecef
    cpnt = np.asarray(cpnt)
    cpnt = cpnt / np.linalg.norm(cpnt)

    # Compute great circle normal
    N = fcross(cpnt, direction) / np.linalg.norm(fcross(cpnt, direction))

    # Create arc points along-track - 1.5 Multiple to have margin for intersection
    max_len = 1.5*circumscription_length(geojson)
    theta = max_len / R_EARTH

    A = rodrigues_rotation(cpnt, N, theta)
    B = rodrigues_rotation(cpnt, N, -theta)

    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)

    # Variables to track maximal cross-track distance perpendicular
    # to along-track arc
    pnt_dist = []

    for pnt in geojson.vertices:
        # Get cross-track point
        C = np.asarray(pnt)
        C = sGEODtoECEF(C, use_degrees=True)
        C = C / np.linalg.norm(C)

        # Get intersection point with arc
        D = fcross(fcross(fcross(A, B), C), fcross(A, B))
        D = D / np.linalg.norm(D)

        # Compute cross-track angle and distance
        ang = math.acos(np.dot(C, D))
        ct_dist = ang * R_EARTH

        # Get directional distance
        dir_dist = np.sign(np.dot(N, D - C)) * ct_dist

        pnt_dist.append(dir_dist)

    # Maximum and minimum indices
    max_idx = np.argmax(pnt_dist)
    min_idx = np.argmin(pnt_dist)

    # Get Separation distance
    dist = pnt_dist[max_idx] - pnt_dist[min_idx]

    return dist, min_idx, pnt_dist[min_idx], max_idx, pnt_dist[max_idx]
