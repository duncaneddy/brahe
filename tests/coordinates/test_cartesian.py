import pytest
import brahe
import numpy as np
from pytest import approx
from brahe import AngleFormat


def test_state_koe_to_eci(eop):
    osc = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    cart = brahe.state_koe_to_eci(osc, AngleFormat.RADIANS)

    assert isinstance(cart, np.ndarray)
    assert cart[0] == brahe.R_EARTH + 500e3
    assert cart[1] == 0.0
    assert cart[2] == 0.0
    assert cart[3] == 0.0
    assert cart[4] == brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.0)
    assert cart[5] == 0.0

    osc = np.array([brahe.R_EARTH + 500e3, 0.0, 90.0, 0.0, 0.0, 0.0])
    cart = brahe.state_koe_to_eci(osc, AngleFormat.DEGREES)

    assert isinstance(cart, np.ndarray)
    assert cart[0] == brahe.R_EARTH + 500e3
    assert cart[1] == 0.0
    assert cart[2] == 0.0
    assert cart[3] == 0.0
    assert cart[4] == pytest.approx(0.0, abs=1.0e-12)
    assert cart[5] == brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.0)


def test_state_eci_to_koe(eop):
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
    osc = brahe.state_eci_to_koe(cart, brahe.AngleFormat.DEGREES)

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
    osc = brahe.state_eci_to_koe(cart, AngleFormat.DEGREES)

    assert osc[0] == approx(brahe.R_EARTH + 500e3, abs=1.0e-9)
    assert osc[1] == 0.0
    assert osc[2] == 90.0
    assert osc[3] == 0.0
    assert osc[4] == 0.0
    assert osc[5] == 0.0


def test_state_eci_to_koe_and_roundtrip(eop):
    osc_original = np.array([brahe.R_EARTH + 700e3, 0.01, 45.0, 120.0, 60.0, 10.0])
    cart = brahe.state_koe_to_eci(osc_original, AngleFormat.DEGREES)
    osc_converted = brahe.state_eci_to_koe(cart, AngleFormat.DEGREES)

    assert osc_converted[0] == approx(osc_original[0], abs=1e-6)
    assert osc_converted[1] == approx(osc_original[1], abs=1e-9)
    assert osc_converted[2] == approx(osc_original[2], abs=1e-6)
    assert osc_converted[3] == approx(osc_original[3], abs=1e-6)
    assert osc_converted[4] == approx(osc_original[4], abs=1e-6)
    assert osc_converted[5] == approx(osc_original[5], abs=1e-6)


def test_state_eci_to_koe_and_roundtri_rad(eop):
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
    cart = brahe.state_koe_to_eci(osc_original, AngleFormat.RADIANS)
    osc_converted = brahe.state_eci_to_koe(cart, AngleFormat.RADIANS)

    assert osc_converted[0] == approx(osc_original[0], abs=1e-6)
    assert osc_converted[1] == approx(osc_original[1], abs=1e-9)
    assert osc_converted[2] == approx(osc_original[2], abs=1e-6)
    assert osc_converted[3] == approx(osc_original[3], abs=1e-6)
    assert osc_converted[4] == approx(osc_original[4], abs=1e-6)
    assert osc_converted[5] == approx(osc_original[5], abs=1e-6)


def test_state_eci_to_koe_for_body_matches_legacy_for_earth_gm(eop):
    """state_eci_to_koe_for_body(x, GM_EARTH, ...) matches state_eci_to_koe exactly."""
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
    osc_legacy = brahe.state_eci_to_koe(cart, AngleFormat.DEGREES)
    osc_for_body = brahe.state_eci_to_koe_for_body(
        cart, brahe.GM_EARTH, AngleFormat.DEGREES
    )

    assert osc_for_body == approx(osc_legacy)


def test_state_eci_to_koe_for_body_lunar_circular_orbit(eop):
    """A circular orbit about the Moon should recover a=r, e=0 using GM_MOON."""
    a = brahe.R_MOON + 100e3
    cart = np.array(
        [a, 0.0, 0.0, 0.0, brahe.periapsis_velocity(a, 0.0, gm=brahe.GM_MOON), 0.0]
    )
    osc = brahe.state_eci_to_koe_for_body(cart, brahe.GM_MOON, AngleFormat.DEGREES)

    assert osc[0] == approx(a, abs=1e-6)
    assert osc[1] == approx(0.0, abs=1e-9)


def test_state_koe_to_eci_for_body_earth_matches_legacy(eop):
    """Mirrors test_state_koe_to_eci_for_body_earth_matches_default."""
    osc = np.array([brahe.R_EARTH + 500e3, 0.01, 97.8, 75.0, 25.0, 45.0])
    via_default = brahe.state_koe_to_eci(osc, AngleFormat.DEGREES)
    via_body = brahe.state_koe_to_eci_for_body(osc, brahe.GM_EARTH, AngleFormat.DEGREES)
    np.testing.assert_array_equal(via_default, via_body)


def test_state_koe_to_eci_for_body_round_trip(eop):
    """Mirrors test_round_trip_conversion_for_body: koe -> eci -> koe about the Moon."""
    osc = np.array([1_838_000.0, 0.01, 85.0, 15.0, 30.0, 45.0])
    cart = brahe.state_koe_to_eci_for_body(osc, brahe.GM_MOON, AngleFormat.DEGREES)
    osc_back = brahe.state_eci_to_koe_for_body(cart, brahe.GM_MOON, AngleFormat.DEGREES)

    assert osc_back[0] == approx(osc[0], abs=1e-8)
    for k in range(1, 6):
        assert osc_back[k] == approx(osc[k], abs=1e-9)
