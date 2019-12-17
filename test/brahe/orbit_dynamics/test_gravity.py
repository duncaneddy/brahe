# Test Imports
import pytest
from pytest import approx
import math
import numpy as np

# Modules Under Test
from brahe.epoch import Epoch
import brahe.constants as _constants
import brahe.frames as _frames
import brahe.ephemerides as _ephem
import brahe.orbit_dynamics.gravity as _grav

def test_load():
    _grav.GravityModel.load(_grav.GRAV_MODEL_EGM2008_90)
    _grav.GravityModel.load(_grav.GRAV_MODEL_GGM05C)
    _grav.GravityModel.load(_grav.GRAV_MODEL_EGM2008_90)

    assert _grav.GravityModel.is_normalized() == True

def test_accel_point_mass(state_itrf):
    # Test Two-Input Acceleration
    r_sat  = np.array([_constants.R_EARTH, 0, 0])
    r_body = np.array([-_constants.R_EARTH, 0, 0])
    
    a_grav = _grav.accel_point_mass(r_sat, r_body)
    assert a_grav[0] == 7.348715716901364
    assert a_grav[1] == 0.0
    assert a_grav[2] == 0.0


    # Single-Input Acceleration
    a_grav = _grav.accel_point_mass(r_sat)
    assert a_grav[0] == approx(-_constants.GM_EARTH/_constants.R_EARTH**2, abs=1e-12)
    assert a_grav[1] == 0.0
    assert a_grav[2] == 0.0

def test_accel_gravity():
    epc = Epoch(2019, 1, 1)
    r_i2b = _frames.rECItoECEF(epc)

    r_sat  = np.array([_constants.R_EARTH, 0, 0])

    with pytest.raises(RuntimeError):
        _grav.accel_gravity(r_sat, r_i2b, 999, 0)

    with pytest.raises(RuntimeError):
        _grav.accel_gravity(r_sat, r_i2b, 0, 999)

    a_grav = _grav.accel_gravity(r_sat, r_i2b, 0, 0)

    assert a_grav[0] == approx(-_constants.GM_EARTH/_constants.R_EARTH**2, abs=1e-12)
    assert a_grav[1] == approx(0.0, abs=1e-12)
    assert a_grav[2] == approx(0.0, abs=1e-12)

    a_grav = _grav.accel_gravity(r_sat, r_i2b, 60, 60)

    assert a_grav[0] == approx(-9.81417404, abs=1e-8)
    assert a_grav[1] == approx(7.99391465e-5, abs=1e-12)
    assert a_grav[2] == approx(-7.98168795e-5, abs=1e-12)

def test_accel_thirdbody(state_gcrf):
    epc = Epoch(2018, 3, 20, 16, 15, 0)

    # Compute moon position
    r_sun = _ephem.sun_position(epc)

    a_sun = _grav.accel_thirdbody(state_gcrf, r_sun, _constants.GM_SUN)

    assert 0.0 < math.fabs(a_sun[0]) < 1.0e-5
    assert 0.0 < math.fabs(a_sun[1]) < 1.0e-5
    assert 0.0 < math.fabs(a_sun[2]) < 1.0e-5

def test_accel_thirdbody_sun(state_gcrf):
    epc = Epoch(2018, 3, 20, 16, 15, 0)

    a_sun = _grav.accel_thirdbody_sun(epc, state_gcrf)

    assert 0.0 < math.fabs(a_sun[0]) < 1.0e-5
    assert 0.0 < math.fabs(a_sun[1]) < 1.0e-5
    assert 0.0 < math.fabs(a_sun[2]) < 1.0e-5

def test_accel_thirdbody_moon(state_gcrf):
    epc = Epoch(2018, 3, 20, 16, 15, 0)

    a_moon = _grav.accel_thirdbody_moon(epc, state_gcrf)

    assert 0.0 < math.fabs(a_moon[0]) < 1.0e-5
    assert 0.0 < math.fabs(a_moon[1]) < 1.0e-5
    assert 0.0 < math.fabs(a_moon[2]) < 1.0e-5