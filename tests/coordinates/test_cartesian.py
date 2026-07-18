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


def test_state_koe_to_inertial_for_body_earth_is_exact(eop):
    """Earth is a bit-identical passthrough of state_koe_to_eci / state_eci_to_koe."""
    osc = np.array([brahe.R_EARTH + 500e3, 0.01, 97.8, 75.0, 25.0, 45.0])
    oracle = brahe.state_koe_to_eci(osc, AngleFormat.DEGREES)
    via_body = brahe.state_koe_to_inertial_for_body(
        osc, brahe.CentralBody.Earth, AngleFormat.DEGREES
    )
    np.testing.assert_array_equal(oracle, via_body)

    osc_back = brahe.state_inertial_to_koe_for_body(
        oracle, brahe.CentralBody.Earth, AngleFormat.DEGREES
    )
    np.testing.assert_array_equal(
        osc_back, brahe.state_eci_to_koe(oracle, AngleFormat.DEGREES)
    )


@pytest.mark.parametrize(
    "central_body,a",
    [
        (brahe.CentralBody.Moon, 1_838_000.0),
        (brahe.CentralBody.Mars, 3_796_000.0),
    ],
)
def test_state_koe_to_inertial_for_body_round_trip(eop, central_body, a):
    """koe -> inertial -> koe about a non-Earth body recovers the input."""
    osc = np.array([a, 0.01, 85.0, 15.0, 30.0, 45.0])
    cart = brahe.state_koe_to_inertial_for_body(osc, central_body, AngleFormat.DEGREES)
    osc_back = brahe.state_inertial_to_koe_for_body(
        cart, central_body, AngleFormat.DEGREES
    )

    assert osc_back[0] == approx(osc[0], abs=1e-8)
    for k in range(1, 6):
        assert osc_back[k] == approx(osc[k], abs=1e-9)


def test_state_koe_to_inertial_for_body_polar_orbit_vs_mars_pole(eop):
    """An i=90 deg orbit referenced to Mars's equator has its orbit normal
    perpendicular to Mars's IAU pole (and not to the ICRF pole)."""
    osc = np.array([brahe.R_MARS + 300e3, 0.0, 90.0, 20.0, 0.0, 0.0])
    cart = brahe.state_koe_to_inertial_for_body(
        osc, brahe.CentralBody.Mars, AngleFormat.DEGREES
    )
    h = np.cross(cart[:3], cart[3:])
    h /= np.linalg.norm(h)

    epc = brahe.Epoch.from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, brahe.TimeSystem.TDB)
    rmat = brahe.rotation_icrf_to_body_fixed_iau(499, epc)
    mars_pole = np.asarray(rmat)[2, :]

    assert np.dot(h, mars_pole) == approx(0.0, abs=1e-12)
    assert abs(h[2]) > 0.1  # not polar relative to the ICRF pole

    osc_back = brahe.state_inertial_to_koe_for_body(
        cart, brahe.CentralBody.Mars, AngleFormat.DEGREES
    )
    assert osc_back[2] == approx(90.0, abs=1e-9)


def test_state_koe_to_inertial_for_body_errors(eop):
    """Barycenters and Custom bodies without a fixed_frame have no mean equator."""
    osc = np.array([3_796_000.0, 0.01, 85.0, 15.0, 30.0, 45.0])
    with pytest.raises(RuntimeError):
        brahe.state_koe_to_inertial_for_body(
            osc, brahe.CentralBody.EMB, AngleFormat.DEGREES
        )
    with pytest.raises(RuntimeError):
        brahe.state_inertial_to_koe_for_body(
            osc, brahe.CentralBody.SSB, AngleFormat.DEGREES
        )

    no_frame = brahe.CentralBody.Custom("Rogue", -99, 1.0e10, radius=1.0e5)
    with pytest.raises(RuntimeError):
        brahe.state_koe_to_inertial_for_body(osc, no_frame, AngleFormat.DEGREES)
