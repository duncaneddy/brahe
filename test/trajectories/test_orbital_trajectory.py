"""
Tests for the OrbitalTrajectory class in brahe.
"""

import pytest
import numpy as np
import brahe


class TestOrbitFrame:
    """Test OrbitFrame enum."""

    def test_eci_frame(self):
        """Test ECI frame creation and properties."""
        eci = brahe.OrbitFrame.eci
        assert str(eci) == "Earth-Centered Inertial (J2000)"
        assert str(eci) == "Earth-Centered Inertial (J2000)"
        assert "ECI" in repr(eci)

    def test_ecef_frame(self):
        """Test ECEF frame creation and properties."""
        ecef = brahe.OrbitFrame.ecef
        assert str(ecef) == "Earth-Centered Earth-Fixed"
        assert str(ecef) == "Earth-Centered Earth-Fixed"
        assert "ECEF" in repr(ecef)

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
        """Test representation creation and properties."""
        cartesian = brahe.OrbitRepresentation.cartesian
        keplerian = brahe.OrbitRepresentation.keplerian

        assert "Cartesian" in str(cartesian)
        assert "Keplerian" in str(keplerian)
        assert cartesian != keplerian

        cartesian2 = brahe.OrbitRepresentation.cartesian
        assert cartesian == cartesian2


class TestAngleFormat:
    """Test AngleFormat enum."""

    def test_angle_formats(self):
        """Test angle format creation and properties."""
        radians = brahe.AngleFormat.radians
        degrees = brahe.AngleFormat.degrees
        none_fmt = brahe.AngleFormat.none

        assert "Radians" in str(radians)
        assert "Degrees" in str(degrees)
        assert "None" in str(none_fmt)

        assert radians != degrees
        assert radians != none_fmt

        radians2 = brahe.AngleFormat.radians
        assert radians == radians2


class TestInterpolationMethod:
    """Test InterpolationMethod enum."""

    def test_interpolation_methods(self):
        """Test interpolation method creation."""
        linear = brahe.InterpolationMethod.linear
        none_interp = brahe.InterpolationMethod.none()

        assert "Linear" in str(linear)
        assert "None" in str(none_interp)
        assert linear != none_interp

        linear2 = brahe.InterpolationMethod.linear
        assert linear == linear2


class TestOrbitalTrajectory:
    """Test OrbitalTrajectory class."""

    def test_empty_orbital_trajectory_creation(self):
        """Test creating an empty orbital trajectory."""
        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        assert len(orbital_traj) == 0
        assert orbital_traj.is_empty()
        assert orbital_traj.frame == brahe.OrbitFrame.eci
        assert orbital_traj.representation == brahe.OrbitRepresentation.cartesian
        assert orbital_traj.angle_format == brahe.AngleFormat.none
        assert "OrbitalTrajectory" in repr(orbital_traj)

    def test_orbital_trajectory_add_state(self):
        """Test adding states to orbital trajectory."""
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
        assert state[0] == pytest.approx(6.678e6, rel=1e-10)
        assert state[4] == pytest.approx(7.726e3, rel=1e-10)

    def test_orbital_trajectory_keplerian_states(self):
        """Test orbital trajectory with Keplerian elements."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.radians,
            brahe.InterpolationMethod.linear
        )

        # ISS-like Keplerian elements [a, e, i, raan, argp, M] in SI units (meters, radians)
        elements = np.array([
            6.778e6,           # semi-major axis (m)
            0.0003,            # eccentricity
            np.radians(51.6),  # inclination (rad)
            np.radians(0.0),   # RAAN (rad)
            np.radians(0.0),   # argument of perigee (rad)
            np.radians(0.0)    # mean anomaly (rad)
        ])

        orbital_traj.add_state(epoch, elements)

        assert len(orbital_traj) == 1
        assert orbital_traj.representation == brahe.OrbitRepresentation.keplerian
        assert orbital_traj.angle_format == brahe.AngleFormat.radians

        # Test state access
        state = orbital_traj.state_at_index(0)
        assert len(state) == 6
        assert state[0] == pytest.approx(6.778e6, rel=1e-10)  # semi-major axis
        assert state[2] == pytest.approx(np.radians(51.6), rel=1e-10)  # inclination

    def test_orbital_trajectory_interpolation(self):
        """Test orbital trajectory interpolation."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        epoch1 = epoch
        epoch2 = epoch + 3600.0  # 1 hour later

        state_vector1 = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        state_vector2 = np.array([6.778e6, 1000e3, 0.0, 100.0, 7.826e3, 50.0])

        orbital_traj.add_state(epoch1, state_vector1)
        orbital_traj.add_state(epoch2, state_vector2)

        # Interpolate at midpoint
        query_epoch = epoch + 1800.0  # 30 minutes later
        interp_state = orbital_traj.state_at_epoch(query_epoch)

        # Verify interpolated values
        expected_x = 6.678e6 + 0.5 * (6.778e6 - 6.678e6)
        expected_y = 0.0 + 0.5 * (1000e3 - 0.0)

        assert interp_state[0] == pytest.approx(expected_x, rel=1e-10)
        assert interp_state[1] == pytest.approx(expected_y, rel=1e-10)

    def test_orbital_trajectory_conversion_to_cartesian(self):
        """Test converting Keplerian orbital trajectory to Cartesian."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create Keplerian trajectory
        kep_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.radians,
            brahe.InterpolationMethod.linear
        )

        # ISS-like Keplerian elements
        elements = np.array([
            6.778e6,           # semi-major axis (m)
            0.0003,            # eccentricity
            np.radians(51.6),  # inclination (rad)
            np.radians(0.0),   # RAAN (rad)
            np.radians(0.0),   # argument of perigee (rad)
            np.radians(0.0)    # mean anomaly (rad)
        ])

        kep_traj.add_state(epoch, elements)

        # Convert to Cartesian
        cart_traj = kep_traj.to_cartesian()

        assert cart_traj.representation == brahe.OrbitRepresentation.cartesian
        assert cart_traj.angle_format == brahe.AngleFormat.none
        assert len(cart_traj) == 1

        # Check that the conversion produces reasonable orbital radius
        cart_state = cart_traj.state_at_index(0)
        position = cart_state[:3]
        radius = np.linalg.norm(position)
        assert 6.6e6 < radius < 7.0e6  # Reasonable orbital radius

    def test_orbital_trajectory_conversion_to_keplerian(self):
        """Test converting Cartesian orbital trajectory to Keplerian."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create Cartesian trajectory
        cart_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # ISS-like orbital state vector
        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        cart_traj.add_state(epoch, state_vector)

        # Convert to Keplerian
        kep_traj = cart_traj.to_keplerian(brahe.AngleFormat.radians)

        assert kep_traj.representation == brahe.OrbitRepresentation.keplerian
        assert kep_traj.angle_format == brahe.AngleFormat.radians
        assert len(kep_traj) == 1

        # Check that semi-major axis is reasonable for ISS-like orbit
        kep_state = kep_traj.state_at_index(0)
        a = kep_state[0]
        assert 6.6e6 < a < 7.0e6  # Semi-major axis in reasonable range

    def test_orbital_trajectory_frame_conversion(self):
        """Test converting between ECI and ECEF frames."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create ECI trajectory
        eci_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # ISS-like orbital state vector
        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        eci_traj.add_state(epoch, state_vector)

        # Convert to ECEF
        ecef_traj = eci_traj.to_frame(brahe.OrbitFrame.ecef)

        assert ecef_traj.frame == brahe.OrbitFrame.ecef
        assert ecef_traj.representation == brahe.OrbitRepresentation.cartesian
        assert len(ecef_traj) == 1

        # Convert back to ECI
        eci_traj2 = ecef_traj.to_frame(brahe.OrbitFrame.eci)

        assert eci_traj2.frame == brahe.OrbitFrame.eci

        # The round-trip conversion should be close to the original
        original_state = eci_traj.state_at_index(0)
        converted_state = eci_traj2.state_at_index(0)

        for i in range(6):
            assert converted_state[i] == pytest.approx(original_state[i], rel=1e-6)

    def test_orbital_trajectory_angle_format_conversion(self):
        """Test angle format conversion for Keplerian trajectories."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create Keplerian trajectory in radians
        rad_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.radians,
            brahe.InterpolationMethod.linear
        )

        elements = np.array([6.778e6, 0.0003, np.radians(51.6), 0.0, 0.0, 0.0])
        rad_traj.add_state(epoch, elements)

        # Convert to degrees
        deg_traj = rad_traj.to_angle_format(brahe.AngleFormat.degrees)

        assert deg_traj.angle_format == brahe.AngleFormat.degrees
        assert deg_traj.representation == brahe.OrbitRepresentation.keplerian

        # Check inclination conversion
        rad_state = rad_traj.state_at_index(0)
        deg_state = deg_traj.state_at_index(0)

        assert deg_state[2] == pytest.approx(np.degrees(rad_state[2]), rel=1e-10)

        # Convert back to radians
        rad_traj2 = deg_traj.to_angle_format(brahe.AngleFormat.radians)
        rad_state2 = rad_traj2.state_at_index(0)

        assert rad_state2[2] == pytest.approx(rad_state[2], rel=1e-10)

    def test_orbital_trajectory_nearest_state(self):
        """Test finding nearest state in orbital trajectory."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # Add multiple states
        epochs = [epoch, epoch + 1800.0, epoch + 3600.0]
        for i, state_epoch in enumerate(epochs):
            state_vector = np.array([6.678e6 + i * 1000, 0.0, 0.0, 0.0, 7.726e3, 0.0])
            orbital_traj.add_state(state_epoch, state_vector)

        # Test nearest_state
        query_epoch = epoch + 900.0  # 15 minutes after first state
        nearest_epoch, nearest_state = orbital_traj.nearest_state(query_epoch)

        assert nearest_epoch.jd() == epochs[0].jd()  # Should be closest to first state
        assert nearest_state[0] == pytest.approx(6.678e6, rel=1e-10)

    def test_orbital_trajectory_multiple_states(self):
        """Test orbital trajectory with multiple states."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # Add states in non-chronological order
        epochs = [epoch + 1800.0, epoch, epoch + 3600.0]
        for i, state_epoch in enumerate(epochs):
            state_vector = np.array([6.678e6 + i * 1000, i * 500, 0.0, 0.0, 7.726e3, i * 10])
            orbital_traj.add_state(state_epoch, state_vector)

        assert len(orbital_traj) == 3

        # Verify states are sorted by epoch
        epoch_0 = orbital_traj.epoch_at_index(0)
        epoch_1 = orbital_traj.epoch_at_index(1)
        epoch_2 = orbital_traj.epoch_at_index(2)

        assert epoch_0.jd() < epoch_1.jd() < epoch_2.jd()

    def test_orbital_trajectory_to_matrix(self):
        """Test orbital trajectory to matrix conversion."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # Add multiple states
        for i in range(3):
            state_epoch = epoch + i * 1800.0
            state_vector = np.array([6.678e6 + i * 1000, i * 500, 0.0, 0.0, 7.726e3, i * 10])
            orbital_traj.add_state(state_epoch, state_vector)

        # Convert to matrix
        matrix = orbital_traj.to_matrix()

        # Should be (6, 3) matrix
        assert matrix.shape == (6, 3)

        # Verify values
        assert matrix[0, 0] == pytest.approx(6.678e6, rel=1e-10)  # First state, x position
        assert matrix[0, 1] == pytest.approx(6.679e6, rel=1e-10)  # Second state, x position
        assert matrix[1, 2] == pytest.approx(1000.0, rel=1e-10)   # Third state, y position

    def test_orbital_trajectory_json_serialization(self):
        """Test JSON serialization and deserialization."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        orbital_traj.add_state(epoch, state_vector)

        # Serialize to JSON
        json_str = orbital_traj.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Deserialize from JSON
        restored_traj = brahe.OrbitalTrajectory.from_json(json_str)

        # Verify restored trajectory matches original
        assert restored_traj.frame == orbital_traj.frame
        assert restored_traj.representation == orbital_traj.representation
        assert restored_traj.angle_format == orbital_traj.angle_format
        assert len(restored_traj) == len(orbital_traj)


class TestOrbitalTrajectoryErrorHandling:
    """Test error handling in OrbitalTrajectory."""

    def test_invalid_state_vector_length(self):
        """Test error for invalid state vector length."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # State vector with wrong length
        invalid_state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3])  # Only 5 elements

        with pytest.raises(ValueError, match="exactly 6 elements"):
            orbital_traj.add_state(epoch, invalid_state)

    def test_orbital_trajectory_empty_interpolation(self):
        """Test orbital trajectory interpolation with empty trajectory."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # Test interpolation with empty trajectory
        with pytest.raises(RuntimeError):
            orbital_traj.state_at_epoch(epoch)

    def test_orbital_trajectory_index_errors(self):
        """Test orbital trajectory indexing errors."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # Add single state
        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        orbital_traj.add_state(epoch, state_vector)

        # Test out of bounds access
        with pytest.raises(IndexError):
            orbital_traj[1]

        with pytest.raises(RuntimeError):
            orbital_traj.state_at_index(1)