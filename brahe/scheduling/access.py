'''The access module provides functionality to compute access opportunities,
geometry, and boundaries under different constraints.
'''

import logging
import typing
import copy
import datetime
import math
import numpy as np

from brahe.constants import RAD2DEG
from brahe.epoch import Epoch
from brahe.tle import TLE
from brahe.coordinates import sECEFtoGEOD
from brahe.astro import orbital_period, sCARTtoOSC

import brahe.data_models as bdm
import brahe.utils as utils
from . import access_geometry as ageo


logger = logging.getLogger(__name__)

###############
# Constraints #
###############

def access_constraints(epc: Epoch, sat_ecef: np.ndarray, loc_ecef: np.ndarray,
                       constraints: bdm.AccessConstraints,
                       constraint_list: typing.List[str], **kwargs):
    '''Check if all access constraints are satisfied.

    Args:
        epc (:obj:`Epoch`): Epoch of geometric access
        sat_ecef (:obj:`np.ndarray`): Satellite ECEF state
        loc_ecef (:obj:`np.ndarray`): Location ECEF
        constraints (:obj:`AccessConstraints`): Constraint object
        constraint_list (List[str]): List of constraint functions to apply to
            check for access.
        kwargs (dict): Accepts arbitrary keywoard arguments 

    Returns:
        bool: Constraints satisfied.
    '''

    valid = True

    for field in constraint_list:
        valid = valid and globals()[f'{field}_constraint'](epc, sat_ecef, loc_ecef, constraints, **kwargs)

    return valid

def look_direction_constraint(epc: Epoch, sat_ecef: np.ndarray,
                              loc_ecef: np.ndarray,
                              constraints: bdm.AccessConstraints, **kwargs):
    '''Look direction access constraint.

    Args:
        epc (:obj:`Epoch`): Epoch of access
        sat_ecef (:obj:`np.ndarray`): Satellite ECEF (ITRF) state
        loc_ecef (:obj:`np.ndarray`): Location ECEF (ITRF) state
        constraints (:obj:`AccessConstraints`): Constraint settings

    Returns:
        bool: True if constraint is satisfied (feasible) at given state and time.
    '''

    look_dir = ageo.look_direction(sat_ecef, loc_ecef)

    if constraints.look_direction == bdm.LookDirection.either or look_dir == constraints.look_direction:
        return True
    else:
        return False


def ascdsc_constraint(epc: Epoch, sat_ecef: np.ndarray, 
                    loc_ecef: np.ndarray, constraints: bdm.AccessConstraints, **kwargs):
    '''Ascending/descending access constraint.

    Args:
        epc (:obj:`Epoch`): Epoch of access
        sat_ecef (:obj:`np.ndarray`): Satellite ECEF (ITRF) state
        loc_ecef (:obj:`np.ndarray`): Location ECEF (ITRF) state
        constraints (:obj:`AccessConstraints`): Constraint settings

    Returns:
        bool: True if constraint is satisfied (feasible) at given state and time.
    '''

    ascdsc = ageo.ascdsc(sat_ecef)

    if constraints.ascdsc == bdm.AscendingDescending.either or ascdsc == constraints.ascdsc:
        return True
    else:
        return False

    return True


def look_angle_constraint(epc: Epoch, sat_ecef: np.ndarray,
                         loc_ecef: np.ndarray,
                         constraints: bdm.AccessConstraints, **kwargs):
    '''Look angle access constraint.

    Args:
        epc (:obj:`Epoch`): Epoch of access
        sat_ecef (:obj:`np.ndarray`): Satellite ECEF (ITRF) state
        loc_ecef (:obj:`np.ndarray`): Location ECEF (ITRF) state
        constraints (:obj:`AccessConstraints`): Constraint settings

    Returns:
        bool: True if constraint is satisfied (feasible) at given state and time.
    '''

    look_angle = ageo.look_angle(sat_ecef, loc_ecef, use_degrees=True)

    if constraints.look_angle_min <= look_angle <= constraints.look_angle_max:
        return True
    else:
        return False


def elevation_constraint(epc: Epoch, sat_ecef: np.ndarray,
                         loc_ecef: np.ndarray,
                         constraints: bdm.AccessConstraints, **kwargs):
    '''Elevation constraint.

    Args:
        epc (:obj:`Epoch`): Epoch of access
        sat_ecef (:obj:`np.ndarray`): Satellite ECEF (ITRF) state
        loc_ecef (:obj:`np.ndarray`): Location ECEF (ITRF) state
        constraints (:obj:`AccessConstraints`): Constraint settings

    Returns:
        bool: True if constraint is satisfied (feasible) at given state and time.
    '''

    azimuth, elevation, _ = ageo.azelrng(sat_ecef, loc_ecef, use_degrees=True)


    if constraints.elevation_min <= elevation <= constraints.elevation_max:
        return True
    else:
        return False


def tile_direction_constraint(epc: Epoch,
                              sat_ecef: np.ndarray,
                              loc_ecef: np.ndarray,
                              constraints: bdm.AccessConstraints,
                              tile: bdm.Tile = None,
                              max_alignment_deviation: float = 10,
                              **kwargs):
    '''Tile direction access constraint. Limits access to satellites 
    aligned with tile direction.

    Args:
        epc (:obj:`Epoch`): Epoch of access
        sat_ecef (:obj:`np.ndarray`): Satellite ECEF (ITRF) state
        loc_ecef (:obj:`np.ndarray`): Location ECEF (ITRF) state
        constraints (:obj:`AccessConstraints`): Constraint settings
        tile (:obj:`tile`): Tile associated with collect
        max_alignment_deviation (float): Maximum deviation of satellite velocity 
            vector and tile direction.

    Returns:
        bool: True if constraint is satisfied (feasible) at given state and time.
    '''

    if len(sat_ecef) < 6:
        raise RuntimeError(
            f'Invalid input length of {len(sat_ecef)}. Must be at least length 6.'
        )

    if not tile:
        raise RuntimeError(f'Missing expected keyword argument "tile"')

    # Satellite Point
    sat_pnt = np.asarray(sat_ecef[0:3])
    sat_pnt = sat_pnt / np.linalg.norm(sat_pnt)

    # Get Direction vectors
    sat_dir = np.asarray(sat_ecef[3:6])
    tile_dir = np.asarray(tile.tile_direction)
    sat_dir = sat_dir / np.linalg.norm(sat_dir)
    tile_dir = tile_dir / np.linalg.norm(tile_dir)

    # Remove component of satellite velocity normal to Earth's surface
    sat_dir = sat_dir - np.dot(sat_dir, sat_pnt)
    sat_dir = sat_dir / np.linalg.norm(sat_dir)

    # Compute alignment of satellite velocity and tile direction
    alignment_vector = np.dot(sat_dir, tile_dir)
    alignment_angle = math.acos(alignment_vector) * RAD2DEG

    if alignment_angle < max_alignment_deviation:
        return True
    else:
        return False

# ######################
# # Access Computation #
# ######################

def find_geometric_constraint_boundary(tle: TLE,
                                       center_ecef: np.ndarray,
                                       constraints: bdm.AccessConstraints,
                                       constraint_list: typing.List[str],
                                       epc0: Epoch,
                                       timestep: float = 1.0,
                                       tol: float = 0.001,
                                       **kwargs) -> Epoch:
    '''Find boundary of next transition from current visibility status (True/False)
    to opposite visibility status (True/False).

    Args:
        tle (:obj:`TLE`): TLE object.
        center_ecef (np.ndarray): Center location to compute access constraints with respect to.
        constraints (typing.List[str]): Access constraint properties.
        constraint_list (List[str]): List of constraint functions to apply to check for access
        timestep (float): timestep for search
        tol (float): Time tolerance for constraint boundaries.
        kwargs (dict): Accepts keyword arguments passed to constraint function

    Returns:
        Epoch: Time boundary (Epoch)
    '''

    # Copy step epoch
    epc = copy.deepcopy(epc0)

    if math.fabs(timestep) < tol:
        return epc0
    else:
        x_ecef = tle.state_itrf(epc)
        visible = access_constraints(epc, x_ecef, center_ecef, constraints, constraint_list, **kwargs)
        while access_constraints(epc, x_ecef, center_ecef, constraints, constraint_list, **kwargs) == visible:
            epc += timestep
            x_ecef = tle.state_itrf(epc)

        next_step = -np.sign(timestep) * max(math.fabs(timestep) / 2.0, tol / 2.0)
        return find_geometric_constraint_boundary(tle, center_ecef, constraints, 
                    constraint_list, epc, timestep=next_step, tol=tol, **kwargs)


def compute_access_properties(tle: TLE, center_ecef: np.ndarray,
                              t_start: Epoch, t_end: Epoch):
    '''Compute access properties of Contact or Collect.

    Args:
        tle (:obj:`TLE`): TLE object
        center_ecef (np.ndarray): Center location to compute access constraints with respect to.
        t_start (:obj:`Epoch`): Start of access window
        t_end (:obj:`Epoch`): End of access window

    Returns:
        AccessProperties: Geometric properties of access
    '''

    # Get Window Midtime
    t_midtime = t_start + (t_end - t_start) / 2.0
    sat_midtime = tle.state_itrf(t_midtime)
    sat_start = tle.state_itrf(t_start)
    sat_end = tle.state_itrf(t_end)

    # Access Properties Object
    access_properties = bdm.AccessProperties()

    # General Geometry
    access_properties.ascdsc = ageo.ascdsc(sat_midtime)
    access_properties.look_direction = ageo.look_direction(sat_midtime, center_ecef)

    # Compute Geometry Properties
    access_properties.azimuth_open = ageo.azimuth(sat_start, center_ecef)
    access_properties.azimuth_close = ageo.azimuth(sat_end, center_ecef)

    # NOTE: Assumes that maximal values for look angle and elevation occur
    # at either the start, end, or midtime.
    access_properties.elevation_min = min(
        ageo.elevation(sat_start, center_ecef), 
        ageo.elevation(sat_end, center_ecef)
    )
    access_properties.elevation_max = ageo.elevation(sat_midtime, center_ecef)

    access_properties.look_angle_min = ageo.look_angle(sat_midtime, center_ecef)
    access_properties.look_angle_max = max(
        ageo.look_angle(sat_start, center_ecef),
        ageo.look_angle(sat_end, center_ecef)
    )

    # Set max and min values
    access_properties.elevation_min = round(access_properties.elevation_min, 6)
    access_properties.elevation_max = round(access_properties.elevation_max, 6)
    access_properties.look_angle_min = round(access_properties.look_angle_min, 6)
    access_properties.look_angle_max = round(access_properties.look_angle_max, 6)

    return access_properties

def find_location_accesses(spacecraft: bdm.Spacecraft, geojson: bdm.GeoJSONObject,
                           t_start: Epoch, t_end: Epoch,
                           timestep: float = 120.0, tol: float = 1e-3,
                           orbit_fraction: float = 0.75, **kwargs):
    '''Final all opportunities for accesses over the period `t_start` to `t_end`. 
    Accepts either `Station` or `Tile` as primary inputs and returns `Contact`
    or `Collect` respectively.

    Args:
        spacecraft (:obj:`Spacecraft`): Spacecraft object.
        geojson (:obj:`Union[Station, Tile]`): Location object with center_point for access. Tile or Station
        t_start (:obj:`Epoch`): Start of window for access computation. GPS Time.
        t_end (:obj:`Epoch`): End of window for access computation. GPS Time.
        timestep (float, Default: 120): timestep for search
        tol (float, Default: 1e-3): Time tolerance for constraint boundaries.
        orbit_fraction (float, Default: 0.75): Fraction of orbital period to advance search after finding an access.
        request (:obj:`Request`): Request. Only required if input GeoJSON is `Tile`
        kwargs (dict): Accepts keyword arguments passed to constraint function.

    Returns:
        List[Union[Contact, Collect]]: `Contact` or `Collect` opportunities.
    '''

    opportunities = []

    # Assert time types as epochs
    t_start = Epoch(t_start, time_system='UTC')
    t_end = Epoch(t_end, time_system='UTC')

    # Set search window based on request window
    if type(geojson) == bdm.Tile:
        request = kwargs.get('request', None)

        if not request:
            raise ValueError(f'Missing kwarg "request"')

    # Get Additional Values
    constraints = None
    if type(geojson) == bdm.Tile:
        constraints = request.constraints

    elif type(geojson) == bdm.Station:
        constraints = geojson.constraints

    else:
        raise ValueError(f'Cannot compute access for geojson input of type {type(geojson)}. Must be Tile or GroundStation.')

    # Set constraint functions to apply
    if type(geojson) == bdm.Tile:
        constraint_list = ['look_direction', 'ascdsc', 'look_angle', 'elevation', 'tile_direction']
    elif type(geojson) == bdm.Station:
        constraint_list = ['elevation']
    else:
        raise ValueError(f'No constraint list defined for geojson input of type {type(geojson)}. Must be Tile or GroundStation.')

    # Add Auxiliary Variables to kwargs
    kwargs['spacecraft_id'] = spacecraft.id
    if type(geojson) == bdm.Tile:
        kwargs['tile'] = geojson

    # Set start time as initial Epoch
    epc = copy.deepcopy(t_start)

    # SGP TLE Propagator
    tle = spacecraft.tle

    # Compute orbital period
    T = orbital_period(sCARTtoOSC(tle.state_gcrf(t_start), use_degrees=True)[0])

    while epc < t_end:

        # Compute satellite state in pseudo-earth-fixed fame
        x_ecef = tle.state_itrf(epc)

        if access_constraints(epc, x_ecef, geojson.center_ecef, constraints, constraint_list, **kwargs):

            # Search for AOS (before initial guess epoch)
            collect_ts = find_geometric_constraint_boundary(tle, geojson.center_ecef,
                constraints, constraint_list, epc, timestep=-timestep, tol=tol, **kwargs
            )

            collect_te = find_geometric_constraint_boundary(tle, geojson.center_ecef,
                constraints, constraint_list, epc, timestep=timestep, tol=tol, **kwargs
            )

            # Create Collect Properties
            if type(geojson) == bdm.Tile:
                # Adjust t_start / t_end based on request properites
                collect_tm = collect_ts + (collect_te - collect_ts)/2.0
                collect_ts = collect_tm - request.properties.collect_duration/2.0
                collect_te = collect_tm + request.properties.collect_duration/2.0

            # Compute Opportunity Properties
            access_properties = compute_access_properties(tle, geojson.center_ecef, collect_ts, collect_te)

            # Create opportunity object
            opportunity = None
            if type(geojson) == bdm.Tile:
                opportunity = bdm.Collect(
                    center=geojson.center.tolist(),
                    center_ecef=geojson.center_ecef.tolist(),
                    t_start=collect_ts.to_datetime(tsys='UTC'),
                    t_end=collect_te.to_datetime(tsys='UTC'),
                    spacecraft_id=spacecraft.id,
                    access_properties=access_properties,
                    tile_id=geojson.tile_id,
                    tile_group_id=geojson.tile_group_id,
                    request_id=request.request_id,
                )

            elif type(geojson) == bdm.Station:
                
                opportunity = bdm.Contact(
                    center=geojson.center.tolist(),
                    center_ecef=geojson.center_ecef.tolist(),
                    t_start=collect_ts.to_datetime(tsys='UTC'),
                    t_end=collect_te.to_datetime(tsys='UTC'),
                    spacecraft_id=spacecraft.id,
                    access_properties=access_properties,
                    station_id=geojson.station_id,
                    station_name=geojson.station_name,
                )

            # Add opportunity to constraints
            opportunities.append(opportunity)

            # Step fraction of an orbit
            epc += orbit_fraction * T
        else:
            epc += timestep

    return opportunities
