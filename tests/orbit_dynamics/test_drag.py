"""
Tests for atmospheric drag acceleration.
"""

import pytest
import numpy as np
import brahe as bh


class TestDrag:
    """Tests for atmospheric drag acceleration."""

    def test_accel_drag(self):
        """Test drag acceleration calculation."""
        # Create orbital elements for 500 km altitude orbit
        oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.3, 15.0, 30.0, 45.0])
        x_object = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)

        density = 1.0e-12  # kg/m³
        mass = 1000.0  # kg
        area = 1.0  # m²
        cd = 2.0  # dimensionless
        T = np.eye(3)  # Identity rotation matrix

        a = bh.accel_drag(x_object, density, mass, area, cd, T)

        assert a.shape == (3,)
        # Drag should oppose motion, so should be non-zero and point generally opposite to velocity
        assert np.linalg.norm(a) > 0.0
        assert np.linalg.norm(a) == pytest.approx(5.97601877277239e-8, abs=1.0e-10)
