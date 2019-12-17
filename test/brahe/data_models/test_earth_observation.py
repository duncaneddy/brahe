# Test Imports
import pytest
from pytest import approx
import uuid

# Modules Under Test
from brahe.epoch import Epoch
from brahe.data_models.earth_observation import *

def test_time_window_properites():
    t_start = Epoch(2020, 1, 1).to_datetime()
    t_mid = Epoch(2020, 1, 1, 12, 0, 0).to_datetime()
    t_end = Epoch(2020, 1, 2).to_datetime()
    t_duration = 86400

    # Initialize from start & end
    twp = TimeWindowProperties(t_start=t_start, t_end=t_end) 
    assert twp.t_duration == 86400 
    assert twp.t_start_epc == Epoch(t_start)
    assert twp.t_end_epc == Epoch(t_end)
    assert twp.t_mid_epc == Epoch(t_mid)

    # Initialize from start & duration
    twp = TimeWindowProperties(t_start=t_start, t_duration=t_duration) 
    assert twp.t_duration == 86400 
    assert twp.t_start_epc == Epoch(t_start)
    assert twp.t_end_epc == Epoch(t_end)
    assert twp.t_mid_epc == Epoch(t_mid)

    # Initialize from end & duration
    twp = TimeWindowProperties(t_end=t_end, t_duration=t_duration) 
    assert twp.t_duration == 86400 
    assert twp.t_start_epc == Epoch(t_start)
    assert twp.t_end_epc == Epoch(t_end)
    assert twp.t_mid_epc == Epoch(t_mid)

def test_time_window_error():
    t_start = Epoch(2020, 1, 1).to_datetime()

    with pytest.raises(ValueError):
        TimeWindowProperties(t_start=t_start)


def test_access_constraints():
    # Check look angle cardinality constraints
    with pytest.raises(ValueError):
        AccessConstraints(look_angle_min=40, look_angle_max=20)

    # Check elevation cardinality constraints
    with pytest.raises(ValueError):
        AccessConstraints(elevation_min=40, elevation_max=20)

def test_request_point():
    point_request = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [-122.401356, 37.770538]
        },
        "properties": {
            "constraints": {
                "ascdsc": "either",
                "look_direction": "either",
                "look_angle_min": 10,
                "look_angle_max": 40,
            },
            "reward": 2
        }
    }

    request = Request(**point_request)

    # Check core GeoJSON
    assert request.type == 'Feature'
    assert request.geotype == 'Point'
    assert len(request.geometry.coordinates) == 2

    # Check added properties
    assert request.geotype == 'Point'
    assert request.reward  == 2

    # Check Derived properties
    assert request.center[0] == -122.401356
    assert request.center[1] == 37.770538

    ecef = request.center_ecef
    assert ecef[0] == approx(-2704991.697152399, abs=1e-8)
    assert ecef[1] == approx(-4262161.912380129, abs=1e-8)
    assert ecef[2] == approx( 3885342.7968954593, abs=1e-8)

    # Check Request properties
    assert request.reward == 2
    assert request.constraints.look_angle_min == 10
    assert request.constraints.look_angle_max == 40
    assert request.constraints.ascdsc.value == 'either'

def test_request_polygon():
    point_request = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
            [
                [-122.6, 37.7], [-122.2,37.7], [-122.2,37.9],
                [-122.6,37.9], [-122.6,37.7]
            ]
            ]
        },
        "properties": {
            "reward": 3,
            "constraints": {
                "ascdsc": "either",
                "look_direction": "either",
                "look_angle_min": 10,
                "look_angle_max": 40,
            }
        }
    }

    request = Request(**point_request)

    # Check core GeoJSON
    assert request.type == 'Feature'
    assert request.geotype == 'Polygon'
    assert request.num_points == 4

    # Check added properties
    assert request.geotype == 'Polygon'

    # Check Derived properties
    assert request.center[0] == approx(-122.4, abs=1e-12)
    assert request.center[1] == approx(37.8, abs=1e-12)

    ecef = request.center_ecef
    assert ecef[0] == approx(-2703817.254689778, abs=1e-8)
    assert ecef[1] == approx(-4260534.252823733, abs=1e-8)
    assert ecef[2] == approx(3887927.165270581, abs=1e-8)

    # Check Request properties
    assert request.reward == 3.0
    assert request.constraints.look_angle_min == 10
    assert request.constraints.look_angle_max == 40
    assert request.constraints.ascdsc.value == 'either'

def test_tile():
    tile_json = {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [-122.6, 37.7], [-122.2,37.7], [-122.2,37.9],
            [-122.6,37.9], [-122.6,37.7]
          ]
        ]
      },
      "properties": {
        "sat_ids": [1],
        "tile_group_id": "1dce1fdf-a7af-4d5e-afb8-70ebc700db79",
        "request_id": '5e365123-7153-40fa-91f9-909b47f7fcb2',
        "tile_direction": [-1, 0, 0]
      }
    }

    tile = Tile(**tile_json)

    # Check core GeoJSON
    assert tile.type == 'Feature'
    assert tile.geotype == 'Polygon'
    assert tile.num_points == 4

    # Check added properties
    assert tile.geotype == 'Polygon'

    # Check Derived properties
    assert tile.center[0] == approx(-122.4, abs=1e-12)
    assert tile.center[1] == approx(37.8, abs=1e-12)

    ecef = tile.center_ecef
    assert ecef[0] == approx(-2703817.254689778, abs=1e-8)
    assert ecef[1] == approx(-4260534.252823733, abs=1e-8)
    assert ecef[2] == approx(3887927.165270581, abs=1e-8)

    # Check Request properties
    assert 1 in tile.sat_ids
    assert tile.request_id == '5e365123-7153-40fa-91f9-909b47f7fcb2'
    assert tile.id != None

def test_station():
    station_json = {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-122.401356, 37.770538]
      },
      "properties": {
        "constraints": {
            "elevation_min": 5
        },
        "downlink_datarate_max": 0
      }
    }

    station = Station(**station_json)

    # Check core GeoJSON
    assert station.type == 'Feature'
    assert station.geotype == 'Point'
    assert len(station.geometry.coordinates) == 2

    # Check added properties
    assert station.geotype == 'Point'

    # Check Derived properties
    assert station.center[0] == -122.401356
    assert station.center[1] == 37.770538

    ecef = station.center_ecef
    assert ecef[0] == approx(-2704991.697152399, abs=1e-8)
    assert ecef[1] == approx(-4262161.912380129, abs=1e-8)
    assert ecef[2] == approx(3885342.7968954593, abs=1e-8)

    # Check Station properties
    assert station.constraints.elevation_min == 10.0
    assert station.downlink_rate_max == 0.0

def test_collect():
    collect_json = {
        "t_start": Epoch(2019, 9, 10, 0, 0, 0).to_datetime(),
        "t_end": Epoch(2019, 9, 11, 0, 0, 0).to_datetime(),
        "center": [-122.401356, 37.770538],
        "spacecraft_id": 1,
        "request_id": '5e365123-7153-40fa-91f9-909b47f7fcb2',
        "tile_id": "23b6d83f-1abb-4f84-91ef-9d40c15311fc",
        "look_angle_min": 30,
        "look_angle_max": 42.0,
        "reward": 4,
    }

    collect = Collect(**collect_json)

    assert collect.t_mid == Epoch(2019, 9, 10, 12, 0, 0).to_datetime()
    assert collect.t_duration == 86400

    assert collect.id != None
    assert collect.spacecraft_id == 1
    assert collect.tile_id == "23b6d83f-1abb-4f84-91ef-9d40c15311fc"
    assert collect.request_id == '5e365123-7153-40fa-91f9-909b47f7fcb2'

    assert collect.center_ecef[0] == approx(-2704991.697152399, abs=1e-8)
    assert collect.center_ecef[1] == approx(-4262161.912380129, abs=1e-8)
    assert collect.center_ecef[2] == approx(3885342.7968954593, abs=1e-8)

def test_contact():
    contact_json = {
        "t_start": Epoch(2019, 9, 10, 0, 0, 0).to_datetime(),
        "t_end": Epoch(2019, 9, 11, 0, 0, 0).to_datetime(),
        "center": [-122.401356, 37.770538],
        "spacecraft_id": 1,
        "station_id": '5e365123-7153-40fa-91f9-909b47f7fcb2',
        "elevation_min": 5.0,
        "elevation_max": 33.0,
        "reward": 4,
    }

    contact = Contact(**contact_json)

    assert contact.t_mid == Epoch(2019, 9, 10, 12, 0, 0).to_datetime()
    assert contact.t_duration == 86400

    assert contact.id != None
    assert contact.spacecraft_id == 1

    assert contact.center_ecef[0] == approx(-2704991.697152399, abs=1e-8)
    assert contact.center_ecef[1] == approx(-4262161.912380129, abs=1e-8)
    assert contact.center_ecef[2] == approx(3885342.7968954593, abs=1e-8)