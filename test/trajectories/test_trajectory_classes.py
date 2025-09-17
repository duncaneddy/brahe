"""
Tests for the trajectory classes (OrbitState, Trajectory, etc.) in brahe.
"""

import pytest
import numpy as np
import brahe


class TestOrbitFrame:
    """Test OrbitFrame enum."""

    def test_eci_frame(self):
        """Test ECI frame creation and properties."""
        eci = brahe.OrbitFrame.eci()
        assert eci.name() == "Earth-Centered Inertial (J2000)"
        assert str(eci) == "Earth-Centered Inertial (J2000)"
        assert "ECI" in repr(eci)

    def test_ecef_frame(self):
        """Test ECEF frame creation and properties."""
        ecef = brahe.OrbitFrame.ecef()
        assert ecef.name() == "Earth-Centered Earth-Fixed"
        assert str(ecef) == "Earth-Centered Earth-Fixed"
        assert "ECEF" in repr(ecef)

    def test_frame_equality(self):
        """Test frame equality comparison."""
        eci1 = brahe.OrbitFrame.eci()
        eci2 = brahe.OrbitFrame.eci()
        ecef = brahe.OrbitFrame.ecef()

        assert eci1 == eci2
        assert eci1 != ecef


class TestOrbitStateType:
    """Test OrbitStateType enum."""

    def test_cartesian_and_keplerian_types(self):
        """Test type creation and properties."""
        cartesian = brahe.OrbitStateType.cartesian()
        keplerian = brahe.OrbitStateType.keplerian()

        assert "Cartesian" in str(cartesian)
        assert "Keplerian" in str(keplerian)
        assert cartesian != keplerian

        cartesian2 = brahe.OrbitStateType.cartesian()
        assert cartesian == cartesian2


class TestAngleFormat:
    """Test AngleFormat enum."""

    def test_angle_formats(self):
        """Test angle format creation and properties."""
        radians = brahe.AngleFormat.radians()
        degrees = brahe.AngleFormat.degrees()
        none_fmt = brahe.AngleFormat.none()

        assert "Radians" in str(radians)
        assert "Degrees" in str(degrees)
        assert "None" in str(none_fmt)

        assert radians != degrees
        assert radians != none_fmt

        radians2 = brahe.AngleFormat.radians()
        assert radians == radians2


class TestInterpolationMethod:
    """Test InterpolationMethod enum."""

    def test_interpolation_methods(self):
        """Test interpolation method creation."""
        linear = brahe.InterpolationMethod.linear()
        none_interp = brahe.InterpolationMethod.none()

        assert "Linear" in str(linear)
        assert "None" in str(none_interp)
        assert linear != none_interp

        linear2 = brahe.InterpolationMethod.linear()
        assert linear == linear2


class TestOrbitState:
    """Test OrbitState class."""

    def test_cartesian_state_creation(self):
        """Test creating a Cartesian orbit state."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # ISS-like orbital state vector [x, y, z, vx, vy, vz] in meters and m/s
        state_vector = np.array([
            6.678e6,   # x position (m)
            0.0,       # y position (m)
            0.0,       # z position (m)
            0.0,       # x velocity (m/s)
            7.726e3,   # y velocity (m/s)
            0.0        # z velocity (m/s)
        ])

        orbit_state = brahe.OrbitState(
            epoch,
            state_vector,
            brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.cartesian(),
            brahe.AngleFormat.none()
        )

        assert orbit_state.epoch().jd() == epoch.jd()
        assert orbit_state.frame == brahe.OrbitFrame.eci()
        assert orbit_state.orbit_type == brahe.OrbitStateType.cartesian()
        assert len(orbit_state) == 6
        assert not orbit_state.is_empty()

    def test_keplerian_state_creation(self):
        """Test creating a Keplerian orbit state."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # ISS-like Keplerian elements [a, e, i, raan, argp, M] in SI units (meters, radians)
        elements = np.array([
            6.778e6,           # semi-major axis (m)
            0.0003,            # eccentricity
            np.radians(51.6),  # inclination (rad)
            np.radians(0.0),   # RAAN (rad)
            np.radians(0.0),   # argument of perigee (rad)
            np.radians(0.0)    # mean anomaly (rad)
        ])

        orbit_state = brahe.OrbitState(
            epoch,
            elements,
            brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.keplerian(),
            brahe.AngleFormat.radians()
        )

        assert orbit_state.epoch().jd() == epoch.jd()
        assert orbit_state.orbit_type == brahe.OrbitStateType.keplerian()
        assert orbit_state.angle_format == brahe.AngleFormat.radians()

    def test_state_vector_access(self):
        """Test accessing state vector elements."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])

        orbit_state = brahe.OrbitState(
            epoch, state_vector, brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
        )

        state = orbit_state.state()
        assert len(state) == 6
        assert state[0] == pytest.approx(6.678e6, rel=1e-6)
        assert state[4] == pytest.approx(7.726e3, rel=1e-6)

        # Test indexing
        assert orbit_state[0] == pytest.approx(6.678e6, rel=1e-6)
        assert orbit_state[4] == pytest.approx(7.726e3, rel=1e-6)

    def test_position_velocity_extraction(self):
        """Test position and velocity extraction from Cartesian state."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])

        orbit_state = brahe.OrbitState(
            epoch, state_vector, brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
        )

        position = orbit_state.position()
        velocity = orbit_state.velocity()

        assert len(position) == 3
        assert len(velocity) == 3
        assert position[0] == pytest.approx(6.678e6, rel=1e-6)
        assert velocity[1] == pytest.approx(7.726e3, rel=1e-6)

    def test_cartesian_to_keplerian_conversion(self):
        """Test converting Cartesian state to Keplerian."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])

        cart_state = brahe.OrbitState(
            epoch, state_vector, brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
        )

        kep_state = cart_state.to_keplerian(brahe.AngleFormat.radians())

        assert kep_state.orbit_type == brahe.OrbitStateType.keplerian()
        assert kep_state.angle_format == brahe.AngleFormat.radians()

        # Check that semi-major axis is reasonable for ISS-like orbit
        elements = kep_state.state()
        a = elements[0]
        assert 6.6e6 < a < 7.0e6  # Semi-major axis in reasonable range

    def test_angle_format_conversion(self):
        """Test angle format conversion."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        elements = np.array([6.778e6, 0.0003, np.radians(51.6), 0.0, 0.0, 0.0])

        rad_state = brahe.OrbitState(
            epoch, elements, brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.keplerian(), brahe.AngleFormat.radians()
        )

        # Original inclination in radians
        original_inclination = rad_state[2]

        # Convert to degrees
        deg_state = rad_state.as_degrees()
        assert deg_state.angle_format == brahe.AngleFormat.degrees()

        deg_inclination = deg_state[2]  # inclination in degrees
        assert deg_inclination == pytest.approx(np.degrees(original_inclination), rel=1e-10)

        # Convert back to radians
        rad_state2 = deg_state.as_radians()
        assert rad_state2.angle_format == brahe.AngleFormat.radians()

        rad_inclination2 = rad_state2[2]
        assert rad_inclination2 == pytest.approx(original_inclination, rel=1e-10)

    def test_metadata(self):
        """Test metadata functionality."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])

        orbit_state = brahe.OrbitState(
            epoch, state_vector, brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
        )

        # Original state should have empty metadata
        metadata = orbit_state.metadata
        assert len(metadata) == 0

        # Add metadata
        new_state = orbit_state.with_metadata("source", "test")
        new_metadata = new_state.metadata
        assert len(new_metadata) == 1
        assert new_metadata["source"] == "test"

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])

        orbit_state = brahe.OrbitState(
            epoch, state_vector, brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
        )

        # Serialize to JSON
        json_str = orbit_state.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Deserialize from JSON
        restored_state = brahe.OrbitState.from_json(json_str)

        # Verify restored state matches original
        assert restored_state.epoch().jd() == orbit_state.epoch().jd()
        assert restored_state.frame == orbit_state.frame
        assert restored_state.orbit_type == orbit_state.orbit_type


class TestTrajectory:
    """Test Trajectory class."""

    def test_empty_trajectory_creation(self):
        """Test creating an empty trajectory."""
        traj = brahe.Trajectory(brahe.InterpolationMethod.linear())

        assert len(traj) == 0
        assert traj.is_empty()
        assert "Trajectory" in repr(traj)
        assert "Linear" in repr(traj)

    def test_trajectory_from_states(self):
        """Test creating trajectory from states."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create multiple states
        states = []
        for i in range(3):
            state_epoch = epoch + i * 1800.0  # 30 minute intervals
            state_vector = np.array([6.678e6 + i * 1000, 0.0, 0.0, 0.0, 7.726e3, 0.0])
            state = brahe.OrbitState(
                state_epoch, state_vector, brahe.OrbitFrame.eci(),
                brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
            )
            states.append(state)

        traj = brahe.Trajectory.from_states(states, brahe.InterpolationMethod.linear())

        assert len(traj) == 3
        assert not traj.is_empty()

        # Test indexing
        first_state = traj[0]
        assert first_state.epoch().jd() == states[0].epoch().jd()

        # Test iteration
        count = 0
        for state in traj:
            assert state.epoch().jd() == states[count].epoch().jd()
            count += 1
        assert count == 3

    def test_trajectory_add_state(self):
        """Test adding states to trajectory."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        traj = brahe.Trajectory(brahe.InterpolationMethod.linear())

        # Add states in non-chronological order
        epochs = [epoch + 1800.0, epoch, epoch + 3600.0]
        for i, state_epoch in enumerate(epochs):
            state_vector = np.array([6.678e6 + i * 1000, 0.0, 0.0, 0.0, 7.726e3, 0.0])
            state = brahe.OrbitState(
                state_epoch, state_vector, brahe.OrbitFrame.eci(),
                brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
            )
            traj.add_state(state)

        assert len(traj) == 3

        # Verify states are sorted by epoch
        assert traj[0].epoch().jd() < traj[1].epoch().jd() < traj[2].epoch().jd()

    def test_trajectory_state_access(self):
        """Test various state access methods."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        traj = brahe.Trajectory(brahe.InterpolationMethod.linear())
        epochs = [epoch, epoch + 1800.0, epoch + 3600.0]

        for i, state_epoch in enumerate(epochs):
            state_vector = np.array([6.678e6 + i * 1000, 0.0, 0.0, 0.0, 7.726e3, 0.0])
            state = brahe.OrbitState(
                state_epoch, state_vector, brahe.OrbitFrame.eci(),
                brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
            )
            traj.add_state(state)

        # Test state_at_index
        state_at_1 = traj.state_at_index(1)
        assert state_at_1.epoch().jd() == epochs[1].jd()

        # Test nearest_state
        query_epoch = epoch + 900.0  # 15 minutes after first state
        nearest_state = traj.nearest_state(query_epoch)
        assert nearest_state.epoch().jd() == epochs[0].jd()  # Should be closest to first state (15 min closer to 0 than 30)

        # Test state_before and state_after
        query_epoch = epoch + 2700.0  # 45 minutes after first state
        before_state = traj.state_before(query_epoch)
        after_state = traj.state_after(query_epoch)

        assert before_state.epoch().jd() == epochs[1].jd()
        assert after_state.epoch().jd() == epochs[2].jd()

    def test_trajectory_interpolation(self):
        """Test trajectory interpolation."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        traj = brahe.Trajectory(brahe.InterpolationMethod.linear())

        epoch1 = epoch
        epoch2 = epoch + 3600.0  # 1 hour later

        state_vector1 = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        state_vector2 = np.array([6.778e6, 1000e3, 0.0, 100.0, 7.826e3, 50.0])

        state1 = brahe.OrbitState(
            epoch1, state_vector1, brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
        )
        state2 = brahe.OrbitState(
            epoch2, state_vector2, brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
        )

        traj.add_state(state1)
        traj.add_state(state2)

        # Interpolate at midpoint
        query_epoch = epoch + 1800.0  # 30 minutes later
        interp_state = traj.state_at_epoch(query_epoch)

        # Verify interpolated values
        interp_vector = interp_state.state()
        expected_x = 6.678e6 + 0.5 * (6.778e6 - 6.678e6)
        expected_y = 0.0 + 0.5 * (1000e3 - 0.0)

        assert interp_vector[0] == pytest.approx(expected_x, rel=1e-10)
        assert interp_vector[1] == pytest.approx(expected_y, rel=1e-10)

    def test_trajectory_to_matrix(self):
        """Test trajectory to matrix conversion."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        traj = brahe.Trajectory(brahe.InterpolationMethod.linear())

        for i in range(3):
            state_epoch = epoch + i * 1800.0
            state_vector = np.array([6.678e6 + i * 1000, i * 500, 0.0, 0.0, 7.726e3, i * 10])
            state = brahe.OrbitState(
                state_epoch, state_vector, brahe.OrbitFrame.eci(),
                brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
            )
            traj.add_state(state)

        # Convert to matrix
        matrix = traj.to_matrix()

        # Should be (6, 3) matrix
        assert matrix.shape == (6, 3)

        # Verify values
        assert matrix[0, 0] == pytest.approx(6.678e6, rel=1e-10)  # First state, x position
        assert matrix[0, 1] == pytest.approx(6.679e6, rel=1e-10)  # Second state, x position
        assert matrix[1, 2] == pytest.approx(1000.0, rel=1e-10)   # Third state, y position


class TestErrorHandling:
    """Test error handling in trajectory classes."""

    def test_invalid_state_vector_length(self):
        """Test error for invalid state vector length."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # State vector with wrong length
        invalid_state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3])  # Only 5 elements

        with pytest.raises(ValueError, match="exactly 6 elements"):
            brahe.OrbitState(
                epoch, invalid_state, brahe.OrbitFrame.eci(),
                brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
            )

    def test_trajectory_empty_interpolation(self):
        """Test trajectory interpolation with empty trajectory."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        traj = brahe.Trajectory(brahe.InterpolationMethod.linear())

        # Test interpolation with empty trajectory
        with pytest.raises(RuntimeError):
            traj.state_at_epoch(epoch)

    def test_trajectory_index_errors(self):
        """Test trajectory indexing errors."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        traj = brahe.Trajectory(brahe.InterpolationMethod.linear())

        # Add single state
        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        state = brahe.OrbitState(
            epoch, state_vector, brahe.OrbitFrame.eci(),
            brahe.OrbitStateType.cartesian(), brahe.AngleFormat.none()
        )
        traj.add_state(state)

        # Test out of bounds access
        with pytest.raises(IndexError):
            traj[1]

        with pytest.raises(RuntimeError):
            traj.state_at_index(1)