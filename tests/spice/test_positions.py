"""Tests for per-body DE velocity/state functions (positions are covered in
tests/orbit_dynamics/test_ephemerides.py)."""

import numpy as np
import pytest

import brahe as bh


@pytest.fixture(autouse=True)
def ensure_kernel():
    try:
        bh.initialize_ephemeris()
    except Exception as e:
        pytest.skip(f"Could not initialize ephemeris: {e}")


BODY_FUNCS = [
    (bh.sun_velocity_de, bh.sun_state_de),
    (bh.moon_velocity_de, bh.moon_state_de),
    (bh.mercury_velocity_de, bh.mercury_state_de),
    (bh.venus_velocity_de, bh.venus_state_de),
    (bh.mars_barycenter_velocity_de, bh.mars_barycenter_state_de),
    (bh.jupiter_barycenter_velocity_de, bh.jupiter_barycenter_state_de),
    (bh.saturn_barycenter_velocity_de, bh.saturn_barycenter_state_de),
    (bh.uranus_barycenter_velocity_de, bh.uranus_barycenter_state_de),
    (bh.neptune_barycenter_velocity_de, bh.neptune_barycenter_state_de),
    (bh.solar_system_barycenter_velocity_de, bh.solar_system_barycenter_state_de),
    (bh.ssb_velocity_de, bh.ssb_state_de),
]


@pytest.mark.parametrize("vel_fn,state_fn", BODY_FUNCS)
def test_velocity_and_state_shapes(vel_fn, state_fn):
    epc = bh.Epoch.from_date(2025, 6, 1, bh.TimeSystem.UTC)
    v = vel_fn(epc, bh.EphemerisSource.DE440s)
    x = state_fn(epc, bh.EphemerisSource.DE440s)
    assert v.shape == (3,)
    assert x.shape == (6,)
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(x[3:], v, atol=1e-9)


def test_sun_velocity_magnitude():
    epc = bh.Epoch.from_date(2025, 6, 1, bh.TimeSystem.UTC)
    v = bh.sun_velocity_de(epc, bh.EphemerisSource.DE440s)
    assert 2.8e4 < np.linalg.norm(v) < 3.1e4


def test_mars_barycenter_position_de():
    # Single-leg barycenter query works with de440s alone (no network).
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r = bh.mars_barycenter_position_de(epc, bh.EphemerisSource.DE440s)
    assert r.shape == (3,)
    assert np.all(np.isfinite(r))


@pytest.mark.integration
def test_mars_position_de_body_center():
    # Downloads mar099s (~64 MB) to compute the Mars body center.
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r_body = bh.mars_position_de(epc, bh.EphemerisSource.DE440s)
    r_bary = bh.mars_barycenter_position_de(epc, bh.EphemerisSource.DE440s)
    # Mars body center differs from the Mars-system barycenter by < 1 km
    # (Phobos/Deimos are tiny) but must be nonzero.
    dr = float(np.linalg.norm(r_body - r_bary))
    assert 0.0 < dr < 1.0e3, f"|body - barycenter| = {dr} m"
    x = bh.mars_state_de(epc, bh.EphemerisSource.DE440s)
    np.testing.assert_allclose(x[:3], r_body, atol=1e-6)
