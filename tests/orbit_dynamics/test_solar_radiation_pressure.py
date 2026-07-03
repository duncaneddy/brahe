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


class TestEclipseConical:
    """Tests for conical (penumbral) eclipse model."""

    def test_eclipse_conical_full_shadow(self):
        """Test conical eclipse with satellite in full shadow (umbra)."""
        # Position satellite on opposite side of Earth from Sun
        r_object = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
        r_sun = np.array([-bh.AU, 0.0, 0.0])

        nu = bh.eclipse_conical(r_object, r_sun)

        # Satellite should be in full shadow
        assert nu == pytest.approx(0.0, abs=1e-12)

    def test_eclipse_conical_full_sunlight(self):
        """Test conical eclipse with satellite in full sunlight."""
        # Position satellite on same side of Earth as Sun
        r_object = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
        r_sun = np.array([bh.AU, 0.0, 0.0])

        nu = bh.eclipse_conical(r_object, r_sun)

        # Satellite should be fully illuminated
        assert nu == pytest.approx(1.0, abs=1e-12)

    def test_eclipse_conical_returns_valid_fraction(self):
        """Test that conical eclipse returns valid illumination fraction."""
        # Test various positions - all should return value between 0 and 1
        r_sun = np.array([-bh.AU, 0.0, 0.0])

        positions = [
            np.array([bh.R_EARTH + 400e3, 0.0, 0.0]),  # In shadow
            np.array([0.0, bh.R_EARTH + 400e3, 0.0]),  # Perpendicular
            np.array([-bh.R_EARTH - 400e3, 0.0, 0.0]),  # Behind Earth
        ]

        for r_object in positions:
            nu = bh.eclipse_conical(r_object, r_sun)
            # All illumination fractions should be valid (between 0 and 1)
            assert 0.0 <= nu <= 1.0

    def test_eclipse_conical_with_state_vector(self):
        """Test conical eclipse with 6D state vector input."""
        r_pos = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
        x_state = np.array([bh.R_EARTH + 400e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
        r_sun = np.array([-bh.AU, 0.0, 0.0])

        # Compute with both inputs
        nu_from_pos = bh.eclipse_conical(r_pos, r_sun)
        nu_from_state = bh.eclipse_conical(x_state, r_sun)

        # Results should be identical (both in full shadow)
        assert nu_from_pos == pytest.approx(nu_from_state, abs=1e-15)
        assert nu_from_pos == pytest.approx(0.0, abs=1e-12)

    def test_eclipse_conical_state_matches_position(self):
        """Test that state and position inputs produce identical results."""
        epc = bh.Epoch.from_date(2024, 1, 1, bh.TimeSystem.UTC)
        r_pos = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
        x_state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
        r_sun = bh.sun_position(epc)

        # Compute with both inputs
        nu_from_pos = bh.eclipse_conical(r_pos, r_sun)
        nu_from_state = bh.eclipse_conical(x_state, r_sun)

        # Results should be identical
        assert nu_from_pos == pytest.approx(nu_from_state, abs=1e-15)


class TestEclipseCylindrical:
    """Tests for cylindrical eclipse model."""

    def test_eclipse_cylindrical_full_shadow(self):
        """Test cylindrical eclipse with satellite in shadow."""
        # Position satellite on opposite side of Earth from Sun
        r_object = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
        r_sun = np.array([-bh.AU, 0.0, 0.0])

        nu = bh.eclipse_cylindrical(r_object, r_sun)

        # Satellite should be in full shadow (cylindrical model is binary)
        assert nu == pytest.approx(0.0, abs=1e-12)

    def test_eclipse_cylindrical_full_sunlight(self):
        """Test cylindrical eclipse with satellite in sunlight."""
        # Position satellite on same side of Earth as Sun
        r_object = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
        r_sun = np.array([bh.AU, 0.0, 0.0])

        nu = bh.eclipse_cylindrical(r_object, r_sun)

        # Satellite should be fully illuminated
        assert nu == pytest.approx(1.0, abs=1e-12)

    def test_eclipse_cylindrical_binary_output(self):
        """Test that cylindrical model returns only 0 or 1."""
        # Test multiple positions
        r_sun = np.array([-bh.AU, 0.0, 0.0])

        # Test various positions
        positions = [
            np.array([bh.R_EARTH + 400e3, 0.0, 0.0]),
            np.array([bh.R_EARTH + 1000e3, 0.0, 0.0]),
            np.array([0.0, bh.R_EARTH + 400e3, 0.0]),
            np.array([0.0, 0.0, bh.R_EARTH + 400e3]),
        ]

        for r_object in positions:
            nu = bh.eclipse_cylindrical(r_object, r_sun)
            # Should be exactly 0.0 or 1.0 (binary)
            assert nu == pytest.approx(0.0, abs=1e-12) or nu == pytest.approx(
                1.0, abs=1e-12
            )

    def test_eclipse_cylindrical_with_state_vector(self):
        """Test cylindrical eclipse with 6D state vector input."""
        r_pos = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
        x_state = np.array([bh.R_EARTH + 400e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
        r_sun = np.array([-bh.AU, 0.0, 0.0])

        # Compute with both inputs
        nu_from_pos = bh.eclipse_cylindrical(r_pos, r_sun)
        nu_from_state = bh.eclipse_cylindrical(x_state, r_sun)

        # Results should be identical (both in full shadow)
        assert nu_from_pos == pytest.approx(nu_from_state, abs=1e-15)
        assert nu_from_pos == pytest.approx(0.0, abs=1e-12)

    def test_eclipse_cylindrical_state_matches_position(self):
        """Test that state and position inputs produce identical results."""
        epc = bh.Epoch.from_date(2024, 1, 1, bh.TimeSystem.UTC)
        r_pos = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
        x_state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
        r_sun = bh.sun_position(epc)

        # Compute with both inputs
        nu_from_pos = bh.eclipse_cylindrical(r_pos, r_sun)
        nu_from_state = bh.eclipse_cylindrical(x_state, r_sun)

        # Results should be identical
        assert nu_from_pos == pytest.approx(nu_from_state, abs=1e-15)


class TestEclipseForBody:
    """Tests for the central-body-aware conical/cylindrical eclipse models."""

    def test_eclipse_for_body_earth_at_origin_matches_legacy(self):
        """eclipse_*_for_body with the occulter at the origin/R_EARTH matches the legacy functions."""
        r_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
        r_sun = np.array([-bh.AU, 0.0, 0.0])
        r_occulter = np.array([0.0, 0.0, 0.0])

        nu_conical_legacy = bh.eclipse_conical(r_object, r_sun)
        nu_conical_for_body = bh.eclipse_conical_for_body(
            r_object, r_sun, r_occulter, bh.R_EARTH
        )
        assert nu_conical_for_body == pytest.approx(nu_conical_legacy)

        nu_cyl_legacy = bh.eclipse_cylindrical(r_object, r_sun)
        nu_cyl_for_body = bh.eclipse_cylindrical_for_body(
            r_object, r_sun, r_occulter, bh.R_EARTH
        )
        assert nu_cyl_for_body == pytest.approx(nu_cyl_legacy)

    def test_eclipse_for_body_offset_occulter(self):
        """Moon occulting an object 2000 km behind it, with the Moon offset from the origin."""
        r_moon = np.array([3.844e8, 0.0, 0.0])
        r_sun = np.array([1.5e11, 0.0, 0.0])

        # Object on the anti-Sun side of the Moon, 2000 km behind it: fully shadowed.
        r_object = np.array([r_moon[0] - 2.0e6, 0.0, 0.0])
        nu = bh.eclipse_cylindrical_for_body(r_object, r_sun, r_moon, bh.R_MOON)
        assert nu == pytest.approx(0.0, abs=1e-12)

        # Object off-axis from the Moon: fully illuminated.
        r_object_free = np.array([r_moon[0], 1.0e7, 0.0])
        nu_free = bh.eclipse_cylindrical_for_body(
            r_object_free, r_sun, r_moon, bh.R_MOON
        )
        assert nu_free == pytest.approx(1.0, abs=1e-12)


class TestEclipseModelsComparison:
    """Tests comparing conical and cylindrical eclipse models."""

    def test_eclipse_models_agreement_in_sunlight(self):
        """Test that both models agree when satellite is fully illuminated."""
        r_object = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
        r_sun = np.array([bh.AU, 0.0, 0.0])

        nu_conical = bh.eclipse_conical(r_object, r_sun)
        nu_cylindrical = bh.eclipse_cylindrical(r_object, r_sun)

        # Both should be 1.0 in full sunlight
        assert nu_conical == pytest.approx(1.0, abs=1e-12)
        assert nu_cylindrical == pytest.approx(1.0, abs=1e-12)

    def test_eclipse_models_agreement_in_umbra(self):
        """Test that both models agree when satellite is in full shadow."""
        r_object = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
        r_sun = np.array([-bh.AU, 0.0, 0.0])

        nu_conical = bh.eclipse_conical(r_object, r_sun)
        nu_cylindrical = bh.eclipse_cylindrical(r_object, r_sun)

        # Both should be 0.0 in full shadow
        assert nu_conical == pytest.approx(0.0, abs=1e-12)
        assert nu_cylindrical == pytest.approx(0.0, abs=1e-12)

    def test_eclipse_models_relationship(self):
        """Test the relationship between conical and cylindrical models."""
        # The cylindrical model is always binary (0 or 1)
        # The conical model can have partial illumination (0 <= nu <= 1)
        # When conical is in partial shadow, cylindrical should be 0
        # Test various positions
        r_sun = np.array([-bh.AU, 0.0, 0.0])

        positions = [
            np.array([bh.R_EARTH + 400e3, 0.0, 0.0]),
            np.array([0.0, bh.R_EARTH + 400e3, 0.0]),
            np.array([-bh.R_EARTH - 400e3, 0.0, 0.0]),
        ]

        for r_object in positions:
            nu_conical = bh.eclipse_conical(r_object, r_sun)
            nu_cylindrical = bh.eclipse_cylindrical(r_object, r_sun)

            # Cylindrical is always binary
            assert nu_cylindrical == pytest.approx(
                0.0, abs=1e-12
            ) or nu_cylindrical == pytest.approx(1.0, abs=1e-12)

            # When conical is partial (penumbra), cylindrical should be 0
            if 0.0 < nu_conical < 1.0:
                assert nu_cylindrical == pytest.approx(0.0, abs=1e-12)

            # When conical is 1.0, cylindrical should be 1.0
            if nu_conical == pytest.approx(1.0, abs=1e-12):
                assert nu_cylindrical == pytest.approx(1.0, abs=1e-12)
