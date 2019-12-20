import pytest
from pytest import approx
import uuid

from brahe.epoch import Epoch

import brahe.data_models as bdm
from brahe.scheduling.tessellation import tessellate
from brahe.scheduling.access import *

# STK Scenario

STK_SVALBARD_ACCESS = [
    (Epoch('2020-01-01T00:19:59.776Z', tsys='UTC'), Epoch('2020-01-01T00:30:39.293Z', tsys='UTC'), 639.517),
    (Epoch('2020-01-01T01:55:16.331Z', tsys='UTC'), Epoch('2020-01-01T02:06:35.211Z', tsys='UTC'), 678.880),
    (Epoch('2020-01-01T03:30:13.201Z', tsys='UTC'), Epoch('2020-01-01T03:41:55.695Z', tsys='UTC'), 702.494),
    (Epoch('2020-01-01T05:04:54.081Z', tsys='UTC'), Epoch('2020-01-01T05:16:30.793Z', tsys='UTC'), 696.712),
    (Epoch('2020-01-01T06:39:19.083Z', tsys='UTC'), Epoch('2020-01-01T06:50:22.980Z', tsys='UTC'), 663.897),
    (Epoch('2020-01-01T08:13:24.935Z', tsys='UTC'), Epoch('2020-01-01T08:23:45.935Z', tsys='UTC'), 621.000),
    (Epoch('2020-01-01T09:47:07.465Z', tsys='UTC'), Epoch('2020-01-01T09:57:01.775Z', tsys='UTC'), 594.310),
    (Epoch('2020-01-01T11:20:29.103Z', tsys='UTC'), Epoch('2020-01-01T11:30:31.919Z', tsys='UTC'), 602.816),
    (Epoch('2020-01-01T12:53:45.755Z', tsys='UTC'), Epoch('2020-01-01T13:04:25.698Z', tsys='UTC'), 639.943),
    (Epoch('2020-01-01T14:27:20.723Z', tsys='UTC'), Epoch('2020-01-01T14:38:41.177Z', tsys='UTC'), 680.454),
    (Epoch('2020-01-01T16:01:32.672Z', tsys='UTC'), Epoch('2020-01-01T16:13:14.011Z', tsys='UTC'), 701.339),
    (Epoch('2020-01-01T17:36:30.019Z', tsys='UTC'), Epoch('2020-01-01T17:48:02.441Z', tsys='UTC'), 692.423),
    (Epoch('2020-01-01T19:12:09.407Z', tsys='UTC'), Epoch('2020-01-01T19:23:08.406Z', tsys='UTC'), 658.999),
    (Epoch('2020-01-01T20:48:14.588Z', tsys='UTC'), Epoch('2020-01-01T20:58:36.775Z', tsys='UTC'), 622.186),
    (Epoch('2020-01-01T22:24:20.109Z', tsys='UTC'), Epoch('2020-01-01T22:34:30.297Z', tsys='UTC'), 610.188),
]

# Tests

def test_geometric_constraints(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    constraint_list = [
        'look_direction', 'ascdsc', 'look_angle', 'elevation'
    ]

    constraints = bdm.AccessConstraints()

    assert access_constraints(epc, sat_ecef, loc_ecef, constraints,
                              constraint_list) == True


def test_look_direction_constraint(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    constraints = bdm.AccessConstraints()

    # Test All Elevations
    constraints.look_direction = bdm.LookDirection.either
    assert look_direction_constraint(epc, sat_ecef, loc_ecef, constraints) == True

    # Test Minimum Elevation Violation
    constraints.look_direction = bdm.LookDirection.left
    assert look_direction_constraint(epc, sat_ecef, loc_ecef, constraints) == True

    # Test Maximum Elevation Violation
    constraints.look_direction = bdm.LookDirection.right
    assert look_direction_constraint(epc, sat_ecef, loc_ecef, constraints) == False


def test_ascdsc_constraint(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    constraints = bdm.AccessConstraints()

    # Test All Elevations
    constraints.ascdsc = bdm.AscendingDescending.either
    assert ascdsc_constraint(epc, sat_ecef, loc_ecef, constraints) == True

    # Test Minimum Elevation Violation
    constraints.ascdsc = bdm.AscendingDescending.ascending
    assert ascdsc_constraint(epc, sat_ecef, loc_ecef, constraints) == True

    # Test Maximum Elevation Violation
    constraints.ascdsc = bdm.AscendingDescending.descending
    assert ascdsc_constraint(epc, sat_ecef, loc_ecef, constraints) == False


def test_look_angle_constraint(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    constraints = bdm.AccessConstraints()

    # Test All Off-Nadir angles
    constraints.look_angle_min = 0.0
    constraints.look_angle_max = 60.0
    assert look_angle_constraint(epc, sat_ecef, loc_ecef, constraints) == True

    # Test Minimum Off-Nadir Violation
    constraints.look_angle_min = 59.0
    constraints.look_angle_max = 60.0
    assert look_angle_constraint(epc, sat_ecef, loc_ecef, constraints) == False

    # Test Maximum Off-Nadir Violation
    constraints.look_angle_min = 0.0
    constraints.look_angle_max = 10.0
    assert look_angle_constraint(epc, sat_ecef, loc_ecef, constraints) == False


def test_elevation_constraint(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    constraints = bdm.AccessConstraints()

    # Test All Elevations
    constraints.elevation_min = 0.0
    constraints.elevation_max = 90.0
    assert elevation_constraint(epc, sat_ecef, loc_ecef, constraints) == True

    # Test Minimum Elevation Violation
    constraints.elevation_min = 89.0
    constraints.elevation_max = 90.0
    assert elevation_constraint(epc, sat_ecef, loc_ecef, constraints) == False

    # Test Maximum Elevation Violation
    constraints.elevation_min = 0.0
    constraints.elevation_max = 45.0
    assert elevation_constraint(epc, sat_ecef, loc_ecef, constraints) == False

def test_access_svalbard_stk(spacecraft_polar, station_svalbard):
    # Compare with output from STK Access

    # Set Duration of Simulation
    t_start = Epoch(2020, 1, 1, time_system='UTC')
    t_end = Epoch(2020, 1, 2, time_system='UTC')

    # Update Elevation to 0 deg
    station_svalbard.properties.constraints.elevation_min = 0.0

    contacts = find_location_accesses(spacecraft_polar, station_svalbard, t_start, t_end)

    assert len(contacts) == 15

    # Check against STK access times:
    for idx, c in enumerate(contacts):
        wo = c.t_start_epc
        wo.time_system = 'UTC'

        wc = c.t_end_epc
        wc.time_system = 'UTC'

        wd = c.t_duration

        assert (wo - STK_SVALBARD_ACCESS[idx][0]) == approx(0, abs=0.05)
        assert (wc - STK_SVALBARD_ACCESS[idx][1]) == approx(0, abs=0.05)
        assert (wd - STK_SVALBARD_ACCESS[idx][2]) == approx(0, abs=0.05)

def test_access_svalbard_stk(spacecraft_polar, request_sf_point):
    # Set Duration of Simulation
    t_start = Epoch(2020, 1, 1, time_system='UTC')
    t_end = Epoch(2020, 1, 8, time_system='UTC')


    # Tile Request
    tiles = tessellate(spacecraft_polar, request_sf_point)

    collects = find_location_accesses(spacecraft_polar, tiles[0], t_start, t_end, request=request_sf_point)

    assert len(collects) == 4