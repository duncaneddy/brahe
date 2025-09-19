"""
Tests for the OrbitalTrajectory class in brahe.

These tests mirror the Rust test suite to ensure Python bindings work correctly.
"""

import pytest
import numpy as np
import brahe


class TestOrbitFrame:
    """Test OrbitFrame enum."""

    def test_eci_frame(self):
        """Test ECI frame creation and properties."""
        eci = brahe.OrbitFrame.eci
        assert "Inertial" in str(eci)

    def test_ecef_frame(self):
        """Test ECEF frame creation and properties."""
        ecef = brahe.OrbitFrame.ecef
        assert "Earth-Fixed" in str(ecef)

    def test_frame_equality(self):
        """Test frame equality comparison."""
        eci1 = brahe.OrbitFrame.eci
        eci2 = brahe.OrbitFrame.eci
        ecef = brahe.OrbitFrame.ecef

        assert eci1 == eci2
        assert eci1 != ecef


class TestOrbitRepresentation:
    """Test OrbitRepresentation enum."""

    def test_cartesian_and_keplerian_types(self):
        """Test Cartesian and Keplerian representation types."""
        cartesian = brahe.OrbitRepresentation.cartesian
        keplerian = brahe.OrbitRepresentation.keplerian

        assert "Cartesian" in str(cartesian)
        assert "Keplerian" in str(keplerian)
        assert cartesian != keplerian


class TestAngleFormat:
    """Test AngleFormat enum."""

    def test_angle_formats(self):
        """Test different angle formats."""
        radians = brahe.AngleFormat.radians
        degrees = brahe.AngleFormat.degrees

        assert "Radians" in str(radians)
        assert "Degrees" in str(degrees)
        assert radians != degrees


class TestInterpolationMethod:
    """Test InterpolationMethod enum."""

    def test_interpolation_methods(self):
        """Test interpolation method creation."""
        linear = brahe.InterpolationMethod.linear
        lagrange_interp = brahe.InterpolationMethod.lagrange

        assert "Linear" in str(linear)
        assert "Lagrange" in str(lagrange_interp)


class TestOrbitalTrajectoryCreation:
    """Test orbital trajectory creation that mirrors Rust tests."""

    def test_orbital_trajectory_creation(self):
        """Test basic orbital trajectory creation."""
        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )
        assert len(orbital_traj) == 0
        assert orbital_traj.is_empty()

    def test_orbital_trajectory_validation(self):
        """Test orbital trajectory with invalid arguments."""
        # This test would check validation if implemented in Python bindings
        # For now, just test that creation works with valid arguments
        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.degrees,
            brahe.InterpolationMethod.linear
        )
        assert len(orbital_traj) == 0


class TestOrbitalTrajectoryStateManagement:
    """Test orbital trajectory state management that mirrors Rust tests."""

    def test_orbital_trajectory_with_data(self):
        """Test orbital trajectory with state data."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # ISS-like orbital state vector [x, y, z, vx, vy, vz] in meters and m/s
        state_vector = np.array([
            6.678e6,   # x position (m)
            0.0,       # y position (m)
            0.0,       # z position (m)
            0.0,       # x velocity (m/s)
            7.726e3,   # y velocity (m/s)
            0.0        # z velocity (m/s)
        ])

        orbital_traj.add_state(epoch, state_vector)

        assert len(orbital_traj) == 1
        assert not orbital_traj.is_empty()

        # Test state access
        state = orbital_traj.state_at_index(0)
        assert len(state) == 6
        np.testing.assert_array_almost_equal(state, state_vector)

    def test_orbital_trajectory_position_velocity(self):
        """Test position and velocity extraction from orbital trajectory."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        state_vector = np.array([1000.0, 2000.0, 3000.0, 1.0, 2.0, 3.0])
        orbital_traj.add_state(epoch, state_vector)

        retrieved_state = orbital_traj.state_at_index(0)
        position = retrieved_state[:3]
        velocity = retrieved_state[3:]

        np.testing.assert_array_almost_equal(position, [1000.0, 2000.0, 3000.0])
        np.testing.assert_array_almost_equal(velocity, [1.0, 2.0, 3.0])


class TestOrbitalTrajectoryConversions:
    """Test orbital trajectory conversions that mirror Rust tests."""

    def test_orbital_trajectory_angle_format_conversion(self):
        """Test angle format conversion."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create trajectory in radians
        rad_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.radians,
            brahe.InterpolationMethod.linear
        )

        # Add a Keplerian state [a, e, i, raan, argp, nu] in radians
        keplerian_state_rad = np.array([
            7000000.0,  # a (m)
            0.01,       # e
            0.872,      # i (radians)
            1.047,      # raan (radians)
            0.524,      # argp (radians)
            0.0         # nu (radians)
        ])

        rad_traj.add_state(epoch, keplerian_state_rad)

        # Convert to degrees
        deg_traj = rad_traj.to_angle_format(brahe.AngleFormat.degrees)

        assert deg_traj.angle_format != rad_traj.angle_format
        assert len(deg_traj) == 1

    def test_orbital_trajectory_frame_conversion(self):
        """Test reference frame conversion."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create ECI trajectory
        eci_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        eci_traj.add_state(epoch, state_vector)

        # Convert to ECEF
        ecef_traj = eci_traj.to_frame(brahe.OrbitFrame.ecef)

        assert ecef_traj.frame != eci_traj.frame
        assert len(ecef_traj) == 1