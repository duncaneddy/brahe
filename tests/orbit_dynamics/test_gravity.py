"""
Tests for gravity acceleration functions.
"""

import pytest
import numpy as np
import brahe as bh


class TestGravity:
    """Tests for gravity acceleration functions."""

    def test_accel_point_mass_gravity(self):
        """Test point-mass gravity acceleration."""
        r_object = np.array([bh.R_EARTH, 0.0, 0.0])
        r_central_body = np.array([0.0, 0.0, 0.0])

        a_grav = bh.accel_point_mass_gravity(r_object, r_central_body, bh.GM_EARTH)

        # Acceleration should be in the negative x-direction
        assert a_grav[0] == pytest.approx(-bh.GM_EARTH / bh.R_EARTH**2, abs=1e-12)
        assert a_grav[1] == pytest.approx(0.0, abs=1e-12)
        assert a_grav[2] == pytest.approx(0.0, abs=1e-12)
        # Magnitude should be roughly -9.8 m/s²
        assert np.linalg.norm(a_grav) == pytest.approx(9.798, abs=1e-3)

    def test_accel_point_mass_gravity_y_axis(self):
        """Test point-mass gravity along y-axis."""
        r_object = np.array([0.0, bh.R_EARTH, 0.0])
        r_central_body = np.array([0.0, 0.0, 0.0])

        a_grav = bh.accel_point_mass_gravity(r_object, r_central_body, bh.GM_EARTH)

        # Acceleration should be in the negative y-direction
        assert a_grav[0] == pytest.approx(0.0, abs=1e-12)
        assert a_grav[1] == pytest.approx(-bh.GM_EARTH / bh.R_EARTH**2, abs=1e-12)
        assert a_grav[2] == pytest.approx(0.0, abs=1e-12)
        assert np.linalg.norm(a_grav) == pytest.approx(9.798, abs=1e-3)

    def test_accel_point_mass_gravity_z_axis(self):
        """Test point-mass gravity along z-axis."""
        r_object = np.array([0.0, 0.0, bh.R_EARTH])
        r_central_body = np.array([0.0, 0.0, 0.0])

        a_grav = bh.accel_point_mass_gravity(r_object, r_central_body, bh.GM_EARTH)

        # Acceleration should be in the negative z-direction
        assert a_grav[0] == pytest.approx(0.0, abs=1e-12)
        assert a_grav[1] == pytest.approx(0.0, abs=1e-12)
        assert a_grav[2] == pytest.approx(-bh.GM_EARTH / bh.R_EARTH**2, abs=1e-12)
        assert np.linalg.norm(a_grav) == pytest.approx(9.798, abs=1e-3)

    def test_accel_point_mass_gravity_with_state_vector(self):
        """Test point-mass gravity with 6D state vector input."""
        # Define state vector [position; velocity]
        x_object = np.array([bh.R_EARTH, 0.0, 0.0, 0.0, 7500.0, 0.0])
        r_central_body = np.array([0.0, 0.0, 0.0])

        # Compute with state vector
        a_grav = bh.accel_point_mass_gravity(x_object, r_central_body, bh.GM_EARTH)

        # Result should be same as using just position
        assert a_grav[0] == pytest.approx(-bh.GM_EARTH / bh.R_EARTH**2, abs=1e-12)
        assert a_grav[1] == pytest.approx(0.0, abs=1e-12)
        assert a_grav[2] == pytest.approx(0.0, abs=1e-12)
        assert np.linalg.norm(a_grav) == pytest.approx(9.798, abs=1e-3)

    def test_accel_point_mass_gravity_state_matches_position(self):
        """Test that state and position inputs produce identical results."""
        r_pos = np.array([bh.R_EARTH + 500e3, 1000e3, 2000e3])
        x_state = np.array([bh.R_EARTH + 500e3, 1000e3, 2000e3, 7500.0, 1000.0, -500.0])
        r_central_body = np.array([0.0, 0.0, 0.0])

        # Compute with both inputs
        a_from_pos = bh.accel_point_mass_gravity(r_pos, r_central_body, bh.GM_EARTH)
        a_from_state = bh.accel_point_mass_gravity(x_state, r_central_body, bh.GM_EARTH)

        # Results should be identical
        assert np.allclose(a_from_pos, a_from_state, atol=1e-15)


class TestSphericalHarmonicGravity:
    """Tests for spherical harmonic gravity models."""

    def test_default_gravity_model_enum(self):
        """Test DefaultGravityModel enum creation and comparison."""
        model1 = bh.DefaultGravityModel.JGM3
        model2 = bh.DefaultGravityModel.JGM3
        model3 = bh.DefaultGravityModel.EGM2008_360

        # Test equality
        assert model1 == model2
        assert model1 != model3

        # Test string representation
        assert "JGM3" in str(model1)
        assert "DefaultGravityModel" in repr(model1)

    def test_gravity_model_tide_system_enum(self):
        """Test GravityModelTideSystem enum."""
        tide1 = bh.GravityModelTideSystem.TideFree
        tide2 = bh.GravityModelTideSystem.ZeroTide

        assert tide1 != tide2
        assert "TideFree" in str(tide1)

    def test_gravity_model_errors_enum(self):
        """Test GravityModelErrors enum."""
        err1 = bh.GravityModelErrors.No
        err2 = bh.GravityModelErrors.Calibrated

        assert err1 != err2
        assert "No" in str(err1)

    def test_gravity_model_normalization_enum(self):
        """Test GravityModelNormalization enum."""
        norm1 = bh.GravityModelNormalization.FullyNormalized
        norm2 = bh.GravityModelNormalization.Unnormalized

        assert norm1 != norm2
        assert "FullyNormalized" in str(norm1)

    def test_gravity_model_from_default_jgm3(self):
        """Test loading JGM3 gravity model."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        assert model.n_max == 70
        assert model.m_max == 70
        assert "JGM3" in model.model_name
        assert model.gm == pytest.approx(bh.GM_EARTH, rel=1e-6)
        assert model.radius == pytest.approx(bh.R_EARTH, rel=1e-6)

        # Test string representations
        assert "JGM3" in str(model)
        assert "70x70" in str(model)
        assert "GravityModel" in repr(model)

    def test_gravity_model_from_default_ggm05s(self):
        """Test loading GGM05S gravity model."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.GGM05S)

        assert model.n_max == 180
        assert model.m_max == 180
        assert "GGM05S" in model.model_name

    def test_gravity_model_from_default_egm2008(self):
        """Test loading EGM2008 gravity model."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.EGM2008_360)

        assert model.n_max == 360
        assert model.m_max == 360
        assert "EGM2008" in model.model_name

    def test_gravity_model_get_coefficients(self):
        """Test retrieving spherical harmonic coefficients."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        # Get J2 coefficient (C20)
        c20, s20 = model.get(2, 0)

        # J2 coefficient should be a negative value around -4.8e-4
        assert c20 == pytest.approx(-4.84169548456e-4, abs=1e-12)
        assert s20 == pytest.approx(0.0, abs=1e-12)

        # Get C21, S21
        c21, s21 = model.get(2, 1)
        assert isinstance(c21, float)
        assert isinstance(s21, float)

    def test_gravity_model_get_invalid_degree(self):
        """Test error handling for invalid degree."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        # Request degree beyond model limits
        with pytest.raises(Exception):
            model.get(100, 0)

    def test_gravity_model_compute_spherical_harmonics(self):
        """Test computing spherical harmonics in body-fixed frame."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        # Test position from Rust tests
        r_body = np.array([6525.919e3, 1710.416e3, 2508.886e3])

        # Compute acceleration with 20x20 expansion
        a_body = model.compute_spherical_harmonics(r_body, 20, 20)

        # Expected values from Rust test (with tolerance)
        assert a_body[0] == pytest.approx(-6.979261862, abs=1e-7)
        assert a_body[1] == pytest.approx(-1.82928315091, abs=1e-7)
        assert a_body[2] == pytest.approx(-2.68999053339, abs=1e-7)

    def test_gravity_model_compute_spherical_harmonics_low_order(self):
        """Test spherical harmonics with low degree/order."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        r_body = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

        # Compute with 2x2 (includes J2)
        a_body = model.compute_spherical_harmonics(r_body, 2, 2)

        # Should be close to point-mass but slightly different due to J2
        expected_mag = bh.GM_EARTH / (bh.R_EARTH + 500e3) ** 2
        actual_mag = np.linalg.norm(a_body)

        # Magnitude should be close to point-mass value
        assert actual_mag == pytest.approx(expected_mag, rel=1e-2)

    def test_gravity_model_compute_invalid_n_max(self):
        """Test error handling for n_max exceeding model limits."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        r_body = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

        # Request degree beyond model limits
        with pytest.raises(Exception):
            model.compute_spherical_harmonics(r_body, 100, 50)

    def test_gravity_model_compute_invalid_m_max(self):
        """Test error handling for m_max > n_max."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        r_body = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

        # Request m_max > n_max (invalid)
        with pytest.raises(Exception):
            model.compute_spherical_harmonics(r_body, 10, 20)

    def test_accel_gravity_spherical_harmonics(self):
        """Test spherical harmonic acceleration in ECI frame."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        # Test position
        r_eci = np.array([6525.919e3, 1710.416e3, 2508.886e3])

        # Use identity matrix (simplified case, body-fixed = ECI)
        R = np.eye(3)

        # Compute acceleration
        a_eci = bh.accel_gravity_spherical_harmonics(r_eci, R, model, 20, 20)

        # With identity rotation, result should match compute_spherical_harmonics
        assert a_eci[0] == pytest.approx(-6.979261862, abs=1e-7)
        assert a_eci[1] == pytest.approx(-1.82928315091, abs=1e-7)
        assert a_eci[2] == pytest.approx(-2.68999053339, abs=1e-7)

    def test_accel_gravity_spherical_harmonics_rotation(self):
        """Test spherical harmonic acceleration with rotation."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        r_eci = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

        # 90-degree rotation about z-axis
        R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        # Compute acceleration
        a_eci = bh.accel_gravity_spherical_harmonics(r_eci, R, model, 10, 10)

        # Acceleration magnitude should be reasonable
        mag = np.linalg.norm(a_eci)
        expected_mag = bh.GM_EARTH / (bh.R_EARTH + 500e3) ** 2
        assert mag == pytest.approx(expected_mag, rel=1e-2)

    def test_accel_gravity_spherical_harmonics_egm2008(self):
        """Test spherical harmonics with high-fidelity EGM2008 model."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.EGM2008_360)

        r_eci = np.array([6525.919e3, 1710.416e3, 2508.886e3])
        R = np.eye(3)

        # Compute with high degree/order
        a_eci = bh.accel_gravity_spherical_harmonics(r_eci, R, model, 50, 50)

        # Should be similar to JGM3 but potentially more accurate
        # Verify magnitude is reasonable
        mag = np.linalg.norm(a_eci)
        expected_mag = bh.GM_EARTH / np.linalg.norm(r_eci) ** 2
        assert mag == pytest.approx(expected_mag, rel=1e-2)

    def test_accel_gravity_spherical_harmonics_with_state_vector(self):
        """Test spherical harmonics with 6D state vector input."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        # Define state vector [position; velocity]
        x_eci = np.array([6525.919e3, 1710.416e3, 2508.886e3, 7500.0, 1000.0, -500.0])
        R = np.eye(3)

        # Compute acceleration
        a_eci = bh.accel_gravity_spherical_harmonics(x_eci, R, model, 20, 20)

        # Should match the result from position-only input
        assert a_eci[0] == pytest.approx(-6.979261862, abs=1e-7)
        assert a_eci[1] == pytest.approx(-1.82928315091, abs=1e-7)
        assert a_eci[2] == pytest.approx(-2.68999053339, abs=1e-7)

    def test_accel_gravity_spherical_harmonics_state_matches_position(self):
        """Test that state and position inputs produce identical results."""
        model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)

        r_pos = np.array([bh.R_EARTH + 500e3, 1000e3, 2000e3])
        x_state = np.array([bh.R_EARTH + 500e3, 1000e3, 2000e3, 7500.0, 1000.0, -500.0])
        R = np.eye(3)

        # Compute with both inputs
        a_from_pos = bh.accel_gravity_spherical_harmonics(r_pos, R, model, 10, 10)
        a_from_state = bh.accel_gravity_spherical_harmonics(x_state, R, model, 10, 10)

        # Results should be identical
        assert np.allclose(a_from_pos, a_from_state, atol=1e-15)
