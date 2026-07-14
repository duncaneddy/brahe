"""Tests for per-body DE velocity/state functions (positions are covered in
tests/orbit_dynamics/test_ephemerides.py)."""

import numpy as np
import pytest

import brahe as bh


@pytest.fixture(autouse=True)
def ensure_kernel():
    try:
        bh.load_spice_kernel("de440s")
    except Exception as e:
        pytest.skip(f"Could not initialize ephemeris: {e}")


BODY_FUNCS = [
    (bh.sun_velocity_spice, bh.sun_state_spice),
    (bh.moon_velocity_spice, bh.moon_state_spice),
    (bh.mercury_velocity_spice, bh.mercury_state_spice),
    (bh.venus_velocity_spice, bh.venus_state_spice),
    (bh.mars_barycenter_velocity_spice, bh.mars_barycenter_state_spice),
    (bh.jupiter_barycenter_velocity_spice, bh.jupiter_barycenter_state_spice),
    (bh.saturn_barycenter_velocity_spice, bh.saturn_barycenter_state_spice),
    (bh.uranus_barycenter_velocity_spice, bh.uranus_barycenter_state_spice),
    (bh.neptune_barycenter_velocity_spice, bh.neptune_barycenter_state_spice),
    (bh.solar_system_barycenter_velocity_spice, bh.solar_system_barycenter_state_spice),
    (bh.ssb_velocity_spice, bh.ssb_state_spice),
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
    v = bh.sun_velocity_spice(epc, bh.EphemerisSource.DE440s)
    assert 2.8e4 < np.linalg.norm(v) < 3.1e4


def test_mars_barycenter_position_spice():
    # Single-leg barycenter query works with de440s alone (no network).
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r = bh.mars_barycenter_position_spice(epc, bh.EphemerisSource.DE440s)
    assert r.shape == (3,)
    assert np.all(np.isfinite(r))


@pytest.mark.integration
def test_mars_position_spice_body_center():
    # Downloads mar099s (~64 MB) to compute the Mars body center.
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r_body = bh.mars_position_spice(epc, bh.EphemerisSource.DE440s)
    r_bary = bh.mars_barycenter_position_spice(epc, bh.EphemerisSource.DE440s)
    # Mars body center differs from the Mars-system barycenter by < 1 km
    # (Phobos/Deimos are tiny) but must be nonzero.
    dr = float(np.linalg.norm(r_body - r_bary))
    assert 0.0 < dr < 1.0e3, f"|body - barycenter| = {dr} m"
    x = bh.mars_state_spice(epc, bh.EphemerisSource.DE440s)
    np.testing.assert_allclose(x[:3], r_body, atol=1e-6)
    v_body = bh.mars_velocity_spice(epc, bh.EphemerisSource.DE440s)
    np.testing.assert_allclose(x[3:], v_body, atol=1e-9)
    # Same two-leg decomposition holds for velocity: body and barycenter
    # velocities differ by a small but nonzero amount.
    v_bary = bh.mars_barycenter_velocity_spice(epc, bh.EphemerisSource.DE440s)
    dv = float(np.linalg.norm(v_body - v_bary))
    assert 0.0 < dv < 1.0, f"|v_body - v_bary| = {dv} m/s"
