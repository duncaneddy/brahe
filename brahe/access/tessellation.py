'''Algorithms for tessellation of geometric areas on Earth
'''

import copy
import logging
import typing
import uuid
import math
import numpy as np

import spherical_geometry.great_circle_arc as sggca
import spherical_geometry.vector as sgv

from . import utils as utils

import brahe.data_models as bdm
from brahe.constants import R_EARTH
from brahe.coordinates import sECEFtoGEOD, sGEODtoECEF

logger = logging.getLogger(__name__)


def create_tile_from_sphere_points(points: typing.List[np.ndarray],
        request: bdm.Request, spacecraft_id: int = None,
        tile_direction: np.ndarray = None) -> bdm.Tile:
    '''Create Tile from list of points.

    Args:
        points (:obj:`points`): Array of points on unit sphere
        request (:obj:`Request`): Request assocaited with tile
        spacecraft_id (int): Spacecraft ID that this tile was generated for.
        tile_direction (:obj:`np.ndarray`): Along-track unit vector direction for tiling

    Returns:
        Tile: Tile object specified by input points.
    '''

    if not spacecraft_id:
        raise RuntimeError('Spacecraft ID required')

    # Convert points
    points = [utils.sphere_point_to_latlon(pnt).tolist() for pnt in points]

    # JSON Template
    tile_json = {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [points]
        },
        'properties': {
            'request_id': request.id,
            'spacecraft_ids': [spacecraft_id],
            'tile_direction': tile_direction.tolist(),
        }
    }

    return bdm.Tile(**tile_json)


def tessellate(spacecraft: bdm.Spacecraft, request: bdm.Request):
    '''Tessellate request object. Calls appropriate tessellation method depending
    on GeoJSON geometry type ("Point" or "Polygon").

    Args:
        spacecraft (:obj:`Spacecraft`): Spacecraft object to use for tessellation.
        request (:obj:`Request`): Request object.

    Returns:
        List[Tile]: Tiles resulting from tessellation.
    '''

    try:
        tiles = globals()[f'tessellate_{request.geotype.lower()}'](spacecraft, request)
    except RuntimeError:
        # Scheduling error here indicates no valid latitue crossing
        logger.error(f'Unable to tessellate {type(request)} - {request.id}')

    return tiles


def tessellate_point(spacecraft: bdm.Spacecraft, request: bdm.Request):
    '''Tessellate request of geometry type "Point".

    Args:
        spacecraft (:obj:`Spacecraft`): Spacecraft object to use for tessellation.
        request (:obj:`Request`): Request object.

    Returns:
        List[Tile]: iles resulting from tessellation. 
    '''

    tiles = []

    # Compute along track directions
    at_dirs = utils.compute_along_track_directions(spacecraft.tle, request.center)

    # Compute strip for ascending and descending
    for at_dir in at_dirs:

        # Spherical Geometry Normalized Center Point
        sgcp = request.center_ecef
        sgcp = np.asarray(sgcp)
        sgcp = sgcp / np.linalg.norm(sgcp)

        # Compute Cross Track Points (Inline with Center)
        ct_ang = request.tessellation.tile_width / R_EARTH

        # Compute Cross-Track Direction
        # ct_dir = np.cross(sgcp, at_dir)

        # Rotate +/- ct_ang/2 using Rodriguez formula
        th = ct_ang / 2.0
        v = sgcp
        k = at_dir
        ct_max = v * math.cos(th) + np.cross(k, v) * math.sin(th) + k * np.dot(k, v) * (1 - math.cos(th))
        ct_min = v * math.cos(-th) + np.cross(k, v) * math.sin(-th) + k * np.dot(k, v) * (1 - math.cos(-th))

        # Compute Along-Track Length
        at_ang = request.tessellation.tile_length / R_EARTH

        # Compute forward strip points
        angle = at_ang / 2.0
        vec = ct_max
        axis = np.cross(ct_min, at_dir)
        pnt1 = utils.rodrigues_rotation(vec, axis, angle)
        vec = ct_min
        axis = np.cross(ct_max, at_dir)
        pnt2 = utils.rodrigues_rotation(vec, axis, angle)

        # Compute reverse Strip Points
        angle = -at_ang / 2.0
        vec = ct_min
        axis = np.cross(ct_min, at_dir)
        pnt3 = utils.rodrigues_rotation(vec, axis, angle)
        vec = ct_max
        axis = np.cross(ct_max, at_dir)
        pnt4 = utils.rodrigues_rotation(vec, axis, angle)

        # Create Tile Object
        tiles.append(
            create_tile_from_sphere_points(
                [pnt1, pnt2, pnt3, pnt4, pnt1],
                request,
                spacecraft_id=spacecraft.id,
                tile_direction=at_dir
            )
        )

    # Return tessellated tiles
    return tiles

def tessellate_polygon(spacecraft: bdm.Spacecraft, request: bdm.Request):
    '''Tessellate request of geometry type "Polygon".

    Args:
        spacecraft (:obj:`Spacecraft`): Spacecraft object to use for tessellation.
        request (:obj:`Request`): Request object.

    Returns:
        List[Tile]: Tessellated tiles.
    '''

    tiles = []

    # Compute along track directions
    at_dirs = utils.compute_along_track_directions(spacecraft.tle, request.center)

    # Tile along each direction
    for at_dir in at_dirs:
        tiles.extend(tile_polygon_direction(request, at_dir, spacecraft_id=spacecraft.spacecraft_id))

    return tiles

def find_polygon_intersection(request: bdm.Request, pnt1: np.ndarray, pnt2: np.ndarray):
    '''Find intesection point of line segment on sphere between two points and
    the input Polygong polygon.

    Args:
        request (:obj:`bdm.Request`): Request to find intersection with
        pnt1 (:obj:`np.ndarray`): Frist point of candidate line segment
        pnt2 (:obj:`np.ndarray`): Second point of candidate line segment

    Returns:
        np.ndarray: Intersecting point.
    '''

    intersection = None

    # Reproject geodetic points onto unit sphere using GEOD -> ECEF transformation
    pnt1 = np.asarray(pnt1)
    pnt1 = sGEODtoECEF(pnt1, use_degrees=True)
    pnt1 = pnt1 / np.linalg.norm(pnt1)

    
    pnt2 = np.asarray(pnt2)
    pnt2 = sGEODtoECEF(pnt2, use_degrees=True)
    pnt2 = pnt2 / np.linalg.norm(pnt2)

    # Iterate through polygon points and test if they interest. 
    for idx in range(0, request.num_points):
        # Create polygon points
        ply1 = request.geometry.coordinates[0][idx]
        ply2 = request.geometry.coordinates[0][idx + 1]

        # Reproject geodetic points of polygon onto unit sphere 
        # Process in done by using GEOD -> ECEF transformation
        ply1 = np.asarray([ply1[0], ply1[1], 0.0])
        ply1 = sGEODtoECEF(ply1, use_degrees=True)
        ply1 = ply1 / np.linalg.norm(ply1)
        ply2 = np.asarray([ply2[0], ply2[1], 0.0])
        ply2 = sGEODtoECEF(ply2, use_degrees=True)
        ply2 = ply2 / np.linalg.norm(ply2)

        # Test Intersection with all polygon points and get "first" one
        # TODO: Should return either all points or "maximal" point.
        if sggca.intersects(pnt1, pnt2, ply1, ply2):

            # Compute intersection point
            N1 = np.cross(pnt1, pnt2)
            N2 = np.cross(ply1, ply2)

            N1 = N1 / np.linalg.norm(N1)
            N2 = N2 / np.linalg.norm(N2)

            # Get Intersection point.
            N3 = np.cross(N1, N2)
            N3 = N3 / np.linalg.norm(N3)

            # Ensure returned point is on correct side of Earth
            cnt = np.asarray(request.center_ecef)
            cnt = cnt / np.linalg.norm(cnt)
            if np.sign(np.dot(cnt, N3)) > 0:
                return N3
            else:
                return -N3

    return intersection


def find_max_alongtrack_distance(request: bdm.Request,
                                 seg1: typing.Tuple[np.ndarray, np.ndarray],
                                 seg2: typing.Tuple[np.ndarray, np.ndarray]):
    '''Find maximum along-track distance for tile. Segements should be the 
    outer and inner lines of of the tile in the along-track. 

    Args:
        request (:obj:bdm.Request): bdm.Request
        seg1 (:obj:Tuple[np.ndarray, np.ndarray]): Outside along-track segment (lon, lat)
        seg2 (:obj:Tuple[np.ndarray, np.ndarray]): Inside along-track segment (lon, lat)

    Returns:
        float: Maximum along-track distance
    '''

    # Check outer segment
    int1 = find_polygon_intersection(request, *seg1)
    l1 = 0.0
    if np.any(int1):
        p = np.asarray(seg1[0])
        p = sGEODtoECEF(p, use_degrees=True)
        p = p / np.linalg.norm(p)

        l1 = math.acos(np.dot(p, int1)) * R_EARTH

    # Check inner segment
    int2 = find_polygon_intersection(request, *seg2)
    l2 = 0.0
    if np.any(int2):
        p = np.asarray(seg2[0])
        p = sGEODtoECEF(p, use_degrees=True)
        p = p / np.linalg.norm(p)

        l2 = math.acos(np.dot(p, int2)) * R_EARTH

    return max(l1, l2)


def create_tile(request: bdm.Request, direction: np.ndarray,
                center_point: np.ndarray, spacecraft_id: int = None):
    '''Create rectangular tile that covers request polygon. Starts with center
    point, rotates 

    Returns:
        List[Tile]: Array of tiles that tessellate along-track direction.
    '''

    tiles = []

    strip_width = request.tessellation.tile_width
    strip_angle = strip_width / R_EARTH

    # Maximal Length
    max_length = utils.circumscription_length(request)
    max_angle = max_length / R_EARTH

    # Get line 1
    ct_pnt_l1 = utils.rodrigues_rotation(center_point, direction, -strip_angle / 2.0)
    ct_geod_l1 = sECEFtoGEOD(ct_pnt_l1 * R_EARTH, use_degrees=True)
    N = np.cross(ct_pnt_l1, direction)
    N = N / np.linalg.norm(N)

    l1_fd = utils.rodrigues_rotation(ct_pnt_l1, N, max_angle)
    l1_bk = utils.rodrigues_rotation(ct_pnt_l1, N, -max_angle)

    # Get line 2
    ct_pnt_l2 = utils.rodrigues_rotation(center_point, direction, strip_angle / 2.0)
    ct_geod_l2 = sECEFtoGEOD(ct_pnt_l2 * R_EARTH, use_degrees=True)
    N = np.cross(ct_pnt_l2, direction)
    N = N / np.linalg.norm(N)

    l2_fd = utils.rodrigues_rotation(ct_pnt_l2, N, max_angle)
    l2_bk = utils.rodrigues_rotation(ct_pnt_l2, N, -max_angle)

    ### Adjust line points to match sphere

    # Max forward distance
    l1_geod_fd = sECEFtoGEOD(l1_fd * R_EARTH, use_degrees=True)
    l2_geod_fd = sECEFtoGEOD(l2_fd * R_EARTH, use_degrees=True)
    lmax = find_max_alongtrack_distance(request, (ct_geod_l1, l1_geod_fd), (ct_geod_l2, l2_geod_fd))

    # Re-adjust forward point distance
    N = np.cross(ct_pnt_l1, direction)
    N = N / np.linalg.norm(N)
    l1_fd = utils.rodrigues_rotation(ct_pnt_l1, N, lmax / R_EARTH)

    N = np.cross(ct_pnt_l2, direction)
    N = N / np.linalg.norm(N)
    l2_fd = utils.rodrigues_rotation(ct_pnt_l2, N, lmax / R_EARTH)

    # Max backward instance
    l1_geod_bk = sECEFtoGEOD(l1_bk * R_EARTH, use_degrees=True)
    l2_geod_bk = sECEFtoGEOD(l2_bk * R_EARTH, use_degrees=True)
    lmin = find_max_alongtrack_distance(request, (ct_geod_l1, l1_geod_bk), (ct_geod_l2, l2_geod_bk))

    # Re-adjust forward point distance
    N = np.cross(ct_pnt_l1, direction)
    N = N / np.linalg.norm(N)
    l1_bk = utils.rodrigues_rotation(ct_pnt_l1, N, -lmin / R_EARTH)

    N = np.cross(ct_pnt_l2, direction)
    N = N / np.linalg.norm(N)
    l2_bk = utils.rodrigues_rotation(ct_pnt_l2, N, -lmin / R_EARTH)

    # Create ile Object
    tile = create_tile_from_sphere_points([l1_fd, l1_bk, l2_bk, l2_fd, l1_fd],
                                          request,
                                          spacecraft_id=spacecraft_id,
                                          tile_direction=direction)

    # Split tiles
    tiles.append(tile)

    return tiles


def tile_polygon_direction(request: bdm.Request, direction: np.ndarray, spacecraft_id: int = None):
    '''Tile request along given direction.

    Args:
        spacecraft (:obj:`Spacecraft`): Spacecraft object to use for tessellation.
        request (:obj:`Request`): Request object.

    Returns:
        List[Tile]: Tessellated tiles.
    '''

    tiles = []

    # Generate Tile Id
    tile_group_id = str(uuid.uuid4())

    ct_width, _, ct_min, _, _ = utils.compute_crosstrack_width(request, direction)

    # Working variables
    tile_width = request.tessellation.tile_width
    ct_overlap = request.tessellation.cross_track_overlap

    # Compute number of required scenes cross-track/along-track
    num_ct = 1
    if ct_width > tile_width:
        num_ct = math.ceil((ct_width - tile_width) / (tile_width - ct_overlap)) + 1

    # Get excess width to center collection
    excess_width = tile_width * num_ct - ct_width

    ct_offset = ct_min + tile_width / 2.0 - (tile_width - ct_overlap) - excess_width / 2.0
    for _ in range(0, num_ct):
        # Compute cross-track point of strip and rotate
        ct_offset += tile_width - ct_overlap
        ct_angle = ct_offset / R_EARTH

        # Rotate Center by angle to get cross-track point
        ct_pnt = utils.rodrigues_rotation(request.center_ecef, direction, ct_angle)
        ct_pnt = ct_pnt / np.linalg.norm(ct_pnt)

        p1, p2 = utils.sphere_point_to_latlon(ct_pnt)[0:2]

        # Compute tiling polygon
        strip_tiles = create_tile(request, direction, ct_pnt, spacecraft_id=spacecraft_id)

        for tile in strip_tiles:
            tile.properties.tile_group_id = tile_group_id

        tiles.extend(strip_tiles)

    return tiles