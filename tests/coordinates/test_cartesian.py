import pytest
import brahe
import numpy as np
from pytest import approx
from brahe import AngleFormat


def test_state_osculating_to_cartesian(eop):
    osc = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    cart = brahe.state_osculating_to_cartesian(osc, AngleFormat.RADIANS)

    assert isinstance(cart, np.ndarray)
    assert cart[0] == brahe.R_EARTH + 500e3
    assert cart[1] == 0.0
    assert cart[2] == 0.0
    assert cart[3] == 0.0
    assert cart[4] == brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.0)
    assert cart[5] == 0.0

    osc = np.array([brahe.R_EARTH + 500e3, 0.0, 90.0, 0.0, 0.0, 0.0])
    cart = brahe.state_osculating_to_cartesian(osc, AngleFormat.DEGREES)

    assert isinstance(cart, np.ndarray)
    assert cart[0] == brahe.R_EARTH + 500e3
    assert cart[1] == 0.0
    assert cart[2] == 0.0
    assert cart[3] == 0.0
    assert cart[4] == pytest.approx(0.0, abs=1.0e-12)
    assert cart[5] == brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.0)


def test_state_cartesian_to_osculating(eop):
    cart = np.array(
        [
            brahe.R_EARTH + 500e3,
            0.0,
            0.0,
            0.0,
            brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.0),
            0.0,
        ]
    )
    osc = brahe.state_cartesian_to_osculating(cart, brahe.AngleFormat.DEGREES)

    assert osc[0] == approx(brahe.R_EARTH + 500e3, abs=1e-9)
    assert osc[1] == 0.0
    assert osc[2] == 0.0
    assert osc[3] == 180.0
    assert osc[4] == 0.0
    assert osc[5] == 0.0

    cart = np.array(
        [
            brahe.R_EARTH + 500e3,
            0.0,
            0.0,
            0.0,
            0.0,
            brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.0),
        ]
    )
    osc = brahe.state_cartesian_to_osculating(cart, AngleFormat.DEGREES)

    assert osc[0] == approx(brahe.R_EARTH + 500e3, abs=1.0e-9)
    assert osc[1] == 0.0
    assert osc[2] == 90.0
    assert osc[3] == 0.0
    assert osc[4] == 0.0
    assert osc[5] == 0.0


def test_state_cartesian_to_osculating_and_roundtrip(eop):
    osc_original = np.array([brahe.R_EARTH + 700e3, 0.01, 45.0, 120.0, 60.0, 10.0])
    cart = brahe.state_osculating_to_cartesian(osc_original, AngleFormat.DEGREES)
    osc_converted = brahe.state_cartesian_to_osculating(cart, AngleFormat.DEGREES)

    assert osc_converted[0] == approx(osc_original[0], abs=1e-6)
    assert osc_converted[1] == approx(osc_original[1], abs=1e-9)
    assert osc_converted[2] == approx(osc_original[2], abs=1e-6)
    assert osc_converted[3] == approx(osc_original[3], abs=1e-6)
    assert osc_converted[4] == approx(osc_original[4], abs=1e-6)
    assert osc_converted[5] == approx(osc_original[5], abs=1e-6)


def test_state_cartesian_to_osculating_and_roundtri_rad(eop):
    osc_original = np.array(
        [
            brahe.R_EARTH + 700e3,
            0.01,
            np.radians(45.0),
            np.radians(120.0),
            np.radians(60.0),
            np.radians(10.0),
        ]
    )
    cart = brahe.state_osculating_to_cartesian(osc_original, AngleFormat.RADIANS)
    osc_converted = brahe.state_cartesian_to_osculating(cart, AngleFormat.RADIANS)

    assert osc_converted[0] == approx(osc_original[0], abs=1e-6)
    assert osc_converted[1] == approx(osc_original[1], abs=1e-9)
    assert osc_converted[2] == approx(osc_original[2], abs=1e-6)
    assert osc_converted[3] == approx(osc_original[3], abs=1e-6)
    assert osc_converted[4] == approx(osc_original[4], abs=1e-6)
    assert osc_converted[5] == approx(osc_original[5], abs=1e-6)
