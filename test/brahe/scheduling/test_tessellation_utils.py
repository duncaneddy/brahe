import pytest
from pytest import approx
import uuid

import spherical_geometry.polygon as sgp

import brahe.cli.cli
from brahe.epoch import Epoch
import brahe.data_models as bdm
import brahe.scheduling.access_geometry as geo
from brahe.scheduling.utils import *


def test_spherical_polygon(request_sf_polygon):
    sp = create_spherical_polygon(request_sf_polygon)

    assert type(sp) == sgp.SphericalPolygon


def test_spherical_polygon_error(request_sf_point):
    with pytest.raises(RuntimeError):
        sp = create_spherical_polygon(request_sf_point)


def test_circumscription_length(request_sf_polygon):
    l = circumscription_length(request_sf_polygon)
    print(request_sf_polygon.center)
    assert l == approx(8207, abs=1)

def test_compute_area(request_sf_polygon):
    area = compute_area(request_sf_polygon)
    assert area == approx(128941405, abs=1)


def test_find_analytic_half_orbits(tle_polar):
    # Helper function to compute argument of latitude
    def arg_lat(tle, epc):
        eci = tle.state_gcrf(epc)
        oe = brahe.sCARTtoOSC(eci, use_degrees=True)
        return (oe[4] + oe[5]) % 360

    # Get Half Orbits
    t_asc, t_dsc = analytical_half_orbit(tle_polar)

    # Check Ascneding is truly ascending
    t_mid = t_asc[0] + (t_asc[1] - t_asc[0]) / 2.0
    assert geo.ascdsc(tle_polar.state_itrf(t_mid)).value == 'ascending'
    assert arg_lat(tle_polar, t_asc[0]) == approx(270.0, abs=0.5)
    assert arg_lat(tle_polar, t_asc[1]) == approx(89.5, abs=0.5)

    # # Check Ascneding is truly ascending
    t_mid = t_dsc[0] + (t_dsc[1] - t_dsc[0]) / 2.0
    assert geo.ascdsc(tle_polar.state_itrf(t_mid)).value == 'descending'
    assert arg_lat(tle_polar, t_dsc[0]) == approx(90.0, abs=0.5)
    assert arg_lat(tle_polar, t_dsc[1]) == approx(270.0, abs=0.5)


def test_find_latitude_crossing(tle_polar):

    # Get Half Orbits
    t_asc, t_dsc = analytical_half_orbit(tle_polar)

    # Get ascending crossing
    asc_epc, asc_ecef = find_latitude_crossing(tle_polar, 37.72, *t_asc)

    assert t_asc[0] <= asc_epc <= t_asc[1]

    # Get ascending crossing
    dsc_epc, dsc_ecef = find_latitude_crossing(tle_polar, 37.72, *t_dsc)

    assert t_dsc[0] <= dsc_epc <= t_dsc[1]


def test_find_latitude_crossing_error(tle_inclined):
    # Get Half Orbits
    t_asc, _ = analytical_half_orbit(tle_inclined)

    # Check non physical crossing
    with pytest.raises(RuntimeError):
        find_latitude_crossing(tle_inclined, 80.0, *t_asc)


def test_compute_along_track_point(tle_polar, request_sf_point):

    # Compute ascending and descending directions
    asc_at, dsc_at = compute_along_track_directions(tle_polar, request_sf_point.center)

    assert np.linalg.norm(asc_at) == pytest.approx(1.0, abs=1e-15)
    assert np.linalg.norm(dsc_at) == pytest.approx(1.0, abs=1e-15)

def test_compute_along_track_polygon(tle_polar, request_sf_polygon):

    # Compute ascending and descending directions
    asc_at, dsc_at = compute_along_track_directions(tle_polar, request_sf_polygon.center)

    assert np.linalg.norm(asc_at) == pytest.approx(1.0, abs=1e-15)
    assert np.linalg.norm(dsc_at) == pytest.approx(1.0, abs=1e-15)

def test_compute_along_track_directions_error(tle_inclined):
    # Create high-latitude request
    request_json = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [0.0, 90.0]
        },
        "properties": {
            "t_start": "2019-03-01T00:00:00.000Z",
            "t_end": "2019-03-08T00:00:00.000Z"
        },
    }

    # Create high lia
    request = bdm.Request(**request_json)

    with pytest.raises(RuntimeError):
        compute_along_track_directions(tle_inclined, request.center)


def test_compute_crosstrack_width(tle_polar, request_sf_polygon):
    # Compute ascending and descending directions
    asc_at, dsc_at = compute_along_track_directions(tle_polar, request_sf_polygon.center)

    # Cross-track width ascending
    dist, min_idx, min_dist, max_idx, max_dist = compute_crosstrack_width(request_sf_polygon, asc_at)
    assert dist == approx(13694, abs=1)
    assert max_dist == approx(6843, abs=1)

    # Cross-track width descending
    dist, min_idx, min_dist, max_idx, max_dist = compute_crosstrack_width(request_sf_polygon, dsc_at)
    assert dist == approx(13686, abs=1)
    assert max_dist == approx(6839, abs=1)
