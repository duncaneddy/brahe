"""
Tests for relativistic acceleration.
"""

import numpy as np
import brahe as bh


class TestRelativity:
    """Tests for relativistic acceleration."""

    def test_accel_relativity(self):
        """Test relativistic acceleration for LEO satellite."""
        # Create orbital elements for 500 km altitude orbit
        oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.3, 15.0, 30.0, 45.0])
        x_object = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

        a = bh.accel_relativity(x_object)

        assert a.shape == (3,)
        # According to Montenbruck and Gill, this should be on order of ~1e-8 for
        # a satellite around 500 km altitude
        assert np.linalg.norm(a) < 1.0e-7
        assert np.linalg.norm(a) > 0.0

    def test_accel_relativity_for_body_matches_legacy_for_earth_gm(self):
        """accel_relativity_for_body(x, GM_EARTH) matches accel_relativity exactly."""
        oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.3, 15.0, 30.0, 45.0])
        x_object = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

        a_legacy = bh.accel_relativity(x_object)
        a_for_body = bh.accel_relativity_for_body(x_object, bh.GM_EARTH)

        assert np.allclose(a_for_body, a_legacy, atol=1e-20)

    def test_accel_relativity_for_body_lunar_gm(self):
        """Relativistic acceleration about the Moon should scale with GM_MOON, not GM_EARTH."""
        x_object = np.array([bh.R_MOON + 100e3, 0.0, 0.0, 0.0, 1600.0, 0.0])

        a_moon = bh.accel_relativity_for_body(x_object, bh.GM_MOON)
        a_earth_gm = bh.accel_relativity_for_body(x_object, bh.GM_EARTH)

        assert a_moon.shape == (3,)
        assert not np.allclose(a_moon, a_earth_gm)
