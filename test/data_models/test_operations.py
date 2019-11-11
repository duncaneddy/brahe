# Test Imports
from pytest import approx
import uuid

# Modules Under Test
from brahe.epoch import Epoch
import brahe.data_models as bdm

def test_request_point():
    point_request = {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-122.401356, 37.770538]
      },
      "properties": {
        "reward": 2,
        "look_angle_min": 10,
        "look_angle_max": 40
      }
    }

    request = bdm.Request(**point_request)

    # Check core GeoJSON
    assert request.type == 'Feature'
    assert request.geometry.type == 'Point'
    assert len(request.geometry.coordinates) == 2

    # Check added properties
    assert request.geotype == 'Point'
    assert request.reward  == 2

    # Check Derived properties
    assert request.center[0] == -122.401356
    assert request.center[1] == 37.770538

    ecef = request.center_ecef
    assert ecef[0] == -2704991.697152399
    assert ecef[1] == -4262161.912380129
    assert ecef[2] == 3885342.7968954593

    # Check Request properties
    assert request.reward == 2
    assert request.look_angle_min == 10
    assert request.look_angle_max == 40
    assert request.ascdsc == bdm.AscendingDescending.either

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
        "look_angle_min": 10,
        "look_angle_max": 40
      }
    }

    request = bdm.Request(**point_request)

    # Check core GeoJSON
    assert request.type == 'Feature'
    assert request.geometry.type == 'Polygon'
    assert request.num_points == 4

    # Check added properties
    assert request.geotype == 'Polygon'

    # Check Derived properties
    assert request.center[0] == approx(-122.4, abs=1e-12)
    assert request.center[1] == approx(37.8, abs=1e-12)

    ecef = request.center_ecef
    assert ecef[0] == -2703817.254689778
    assert ecef[1] == -4260534.252823733
    assert ecef[2] == 3887927.165270581

    # Check Request properties
    assert request.reward == 3.0
    assert request.look_angle_min == 10
    assert request.look_angle_max == 40
    assert request.ascdsc == bdm.AscendingDescending.either

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
        "request_id": 3,
        "sat_ids": [1],
        "ascdsc": "descending"
      }
    }

    tile = bdm.Tile(**tile_json)

    # Check core GeoJSON
    assert tile.type == 'Feature'
    assert tile.geometry.type == 'Polygon'
    assert tile.num_points == 4

    # Check added properties
    assert tile.geotype == 'Polygon'

    # Check Derived properties
    assert tile.center[0] == approx(-122.4, abs=1e-12)
    assert tile.center[1] == approx(37.8, abs=1e-12)

    ecef = tile.center_ecef
    assert ecef[0] == -2703817.254689778
    assert ecef[1] == -4260534.252823733
    assert ecef[2] == 3887927.165270581

    # Check Request properties
    assert 1 in tile.sat_ids
    assert tile.request_id == 3
    assert tile.id != None
    assert tile.ascdsc == bdm.AscendingDescending.descending

def test_station():
    station_json = {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-122.401356, 37.770538]
      },
      "properties": {
        "elevation_min": 10,
      }
    }

    station = bdm.Station(**station_json)

    # Check core GeoJSON
    assert station.type == 'Feature'
    assert station.geometry.type == 'Point'
    assert len(station.geometry.coordinates) == 2

    # Check added properties
    assert station.geotype == 'Point'

    # Check Derived properties
    assert station.center[0] == -122.401356
    assert station.center[1] == 37.770538

    ecef = station.center_ecef
    assert ecef[0] == -2704991.697152399
    assert ecef[1] == -4262161.912380129
    assert ecef[2] == 3885342.7968954593

    # Check Station properties
    assert station.elevation_min == 10.0
    assert station.cost_per_min == 0.0
    assert station.downlink_datarate == 0.0

def test_collect():
    collect_json = {
        "t_start": Epoch(2019, 9, 10, 0, 0, 0).to_datetime(),
        "t_end": Epoch(2019, 9, 11, 0, 0, 0).to_datetime(),
        "center": [-122.401356, 37.770538],
        "sat_id": 1,
        "request_id": 99,
        "tile_id": "23b6d83f-1abb-4f84-91ef-9d40c15311fc",
        "look_angle_min": 30,
        "look_angle_max": 42.0,
        "reward": 4,
    }

    collect = bdm.Collect(**collect_json)

    assert collect.t_mid == Epoch(2019, 9, 10, 12, 0, 0).to_datetime()
    assert collect.t_duration == 86400

    assert collect.id != None
    assert collect.sat_id == 1
    assert collect.tile_id == uuid.UUID("23b6d83f-1abb-4f84-91ef-9d40c15311fc")
    assert collect.request_id == 99

    assert collect.center_ecef[0] == -2704991.697152399
    assert collect.center_ecef[1] == -4262161.912380129
    assert collect.center_ecef[2] == 3885342.7968954593

def test_contact():
    contact_json = {
        "t_start": Epoch(2019, 9, 10, 0, 0, 0).to_datetime(),
        "t_end": Epoch(2019, 9, 11, 0, 0, 0).to_datetime(),
        "center": [-122.401356, 37.770538],
        "sat_id": 1,
        "station_id": 2,
        "elevation_min": 5.0,
        "elevation_max": 33.0,
        "reward": 4,
    }

    contact = bdm.Contact(**contact_json)

    assert contact.t_mid == Epoch(2019, 9, 10, 12, 0, 0).to_datetime()
    assert contact.t_duration == 86400

    assert contact.id != None
    assert contact.sat_id == 1

    assert contact.center_ecef[0] == -2704991.697152399
    assert contact.center_ecef[1] == -4262161.912380129
    assert contact.center_ecef[2] == 3885342.7968954593