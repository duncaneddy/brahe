import pytest
from pytest import approx
import uuid

from brahe.epoch import Epoch

import brahe.data_models as bdm
from brahe.scheduling.access_geometry import *


def test_azelrng(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    az, el, rn = azelrng(sat_ecef, loc_ecef, use_degrees=True)

    assert az == approx(89.99, abs=1e-1)
    assert el == approx(66.11, abs=1e-2)
    assert rn == approx(596939, abs=1.0)


def test_azimuth(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    az = azimuth(sat_ecef, loc_ecef, use_degrees=True)

    assert az == approx(89.99, abs=1e-1)


def test_elevation(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    el = elevation(sat_ecef, loc_ecef, use_degrees=True)

    assert el == approx(66.11, abs=1e-2)


def test_range(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    rn = range(sat_ecef, loc_ecef, use_degrees=True)

    assert rn == approx(596939, abs=1.0)


def test_look_angle(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    th = look_angle(sat_ecef, loc_ecef, use_degrees=True)

    assert th == approx(21.89, abs=1e-2)


def test_ascdsc_asc(access_geometry_left):
    _, sat_ecef, _ = access_geometry_left

    acdc = ascdsc(sat_ecef)

    assert acdc.value == 'ascending'


def test_ascdsc_dsc(access_geometry_right):
    _, sat_ecef, _ = access_geometry_right

    acdc = ascdsc(sat_ecef)

    assert acdc.value == 'descending'


def test_look_direction_left(access_geometry_left):
    epc, sat_ecef, loc_ecef = access_geometry_left

    ld = look_direction(sat_ecef, loc_ecef)

    assert ld.value == 'left'


def test_look_direction_right(access_geometry_right):
    epc, sat_ecef, loc_ecef = access_geometry_right

    ld = look_direction(sat_ecef, loc_ecef)

    assert ld.value == 'right'
