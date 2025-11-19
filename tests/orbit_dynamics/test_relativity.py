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
        x_object = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)

        a = bh.accel_relativity(x_object)

        assert a.shape == (3,)
        # According to Montenbruck and Gill, this should be on order of ~1e-8 for
        # a satellite around 500 km altitude
        assert np.linalg.norm(a) < 1.0e-7
        assert np.linalg.norm(a) > 0.0
