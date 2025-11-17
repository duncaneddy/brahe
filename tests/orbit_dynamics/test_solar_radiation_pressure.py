"""
Tests for solar radiation pressure acceleration.
"""

import pytest
import numpy as np
import brahe as bh


class TestSolarRadiationPressure:
    """Tests for solar radiation pressure acceleration."""

    def test_accel_solar_radiation_pressure(self):
        """Test SRP acceleration at 1 AU."""
        r_object = np.array([bh.AU, 0.0, 0.0])
        r_sun = np.array([0.0, 0.0, 0.0])

        mass = 1.0  # kg
        cr = 1.0  # dimensionless
        area = 1.0  # m²
        p0 = 4.5e-6  # N/m²

        a_srp = bh.accel_solar_radiation_pressure(r_object, r_sun, mass, cr, area, p0)

        # Acceleration should be in the positive x-direction at 1 AU
        assert a_srp[0] == pytest.approx(4.5e-6, abs=1e-12)
        assert a_srp[1] == pytest.approx(0.0, abs=1e-12)
        assert a_srp[2] == pytest.approx(0.0, abs=1e-12)

    def test_accel_solar_radiation_pressure_with_sun_position(self):
        """Test SRP with realistic sun position."""
        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
        r_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
        r_sun = bh.sun_position(epc)

        mass = 1000.0  # kg
        cr = 1.8  # dimensionless
        area = 1.0  # m²
        p0 = 4.56e-6  # N/m²

        a_srp = bh.accel_solar_radiation_pressure(r_object, r_sun, mass, cr, area, p0)

        assert a_srp.shape == (3,)
        # SRP should be non-zero but small at LEO
        assert np.linalg.norm(a_srp) > 0.0

    def test_accel_solar_radiation_pressure_with_state_vector(self):
        """Test SRP acceleration with 6D state vector input."""
        r_pos = np.array([bh.AU, 0.0, 0.0])
        x_state = np.array([bh.AU, 0.0, 0.0, 0.0, 30000.0, 0.0])
        r_sun = np.array([0.0, 0.0, 0.0])

        mass = 1.0  # kg
        cr = 1.0  # dimensionless
        area = 1.0  # m²
        p0 = 4.5e-6  # N/m²

        # Compute with both inputs
        a_from_pos = bh.accel_solar_radiation_pressure(r_pos, r_sun, mass, cr, area, p0)
        a_from_state = bh.accel_solar_radiation_pressure(
            x_state, r_sun, mass, cr, area, p0
        )

        # Results should be identical
        assert np.allclose(a_from_pos, a_from_state, atol=1e-15)

    def test_accel_solar_radiation_pressure_state_matches_position(self):
        """Test that state and position inputs produce identical results with realistic sun."""
        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
        r_pos = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
        x_state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
        r_sun = bh.sun_position(epc)

        mass = 1000.0  # kg
        cr = 1.8  # dimensionless
        area = 1.0  # m²
        p0 = 4.56e-6  # N/m²

        # Compute with both inputs
        a_from_pos = bh.accel_solar_radiation_pressure(r_pos, r_sun, mass, cr, area, p0)
        a_from_state = bh.accel_solar_radiation_pressure(
            x_state, r_sun, mass, cr, area, p0
        )

        # Results should be identical
        assert np.allclose(a_from_pos, a_from_state, atol=1e-15)
