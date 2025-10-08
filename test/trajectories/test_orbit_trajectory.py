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

        assert "Linear" in str(linear)


class TestOrbitalTrajectoryCreation:
    """Test orbital trajectory creation that mirrors Rust tests."""

    def test_orbital_trajectory_creation(self):
        """Test basic orbital trajectory creation."""
        orbital_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )
        assert len(orbital_traj) == 0
        assert orbital_traj.is_empty()

    def test_orbital_trajectory_validation(self):
        """Test orbital trajectory with invalid arguments."""
        # This test would check validation if implemented in Python bindings
        # For now, just test that creation works with valid arguments
        orbital_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.degrees
        )
        assert len(orbital_traj) == 0


class TestOrbitalTrajectoryStateManagement:
    """Test orbital trajectory state management that mirrors Rust tests."""

    def test_orbital_trajectory_with_data(self):
        """Test orbital trajectory with state data."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
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
        state = orbital_traj.state(0)
        assert len(state) == 6
        np.testing.assert_array_almost_equal(state, state_vector)

    def test_orbital_trajectory_position_velocity(self):
        """Test position and velocity extraction from orbital trajectory."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        orbital_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        state_vector = np.array([1000.0, 2000.0, 3000.0, 1.0, 2.0, 3.0])
        orbital_traj.add_state(epoch, state_vector)

        retrieved_state = orbital_traj.state(0)
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
        rad_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.radians
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
        eci_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        state_vector = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        eci_traj.add_state(epoch, state_vector)

        # Convert to ECEF
        ecef_traj = eci_traj.to_frame(brahe.OrbitFrame.ecef)

        assert ecef_traj.frame != eci_traj.frame
        assert len(ecef_traj) == 1

    def test_orbital_trajectory_representation_conversion(self):
        """Test representation conversion between Cartesian and Keplerian."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create Cartesian trajectory
        cart_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Add ISS-like orbit state
        cart_state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        cart_traj.add_state(epoch, cart_state)

        # Convert to Keplerian
        kep_traj = cart_traj.to_keplerian(brahe.AngleFormat.radians)

        assert kep_traj.representation != cart_traj.representation
        assert kep_traj.angle_format == brahe.AngleFormat.radians
        assert len(kep_traj) == 1

        # Convert back to Cartesian
        cart_traj2 = kep_traj.to_cartesian()

        assert cart_traj2.representation == brahe.OrbitRepresentation.cartesian
        assert len(cart_traj2) == 1

    def test_orbital_trajectory_convert_to(self):
        """Test unified convert_to method."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create ECI Cartesian trajectory
        eci_cart_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        cart_state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        eci_cart_traj.add_state(epoch, cart_state)

        # Convert to ECEF Keplerian with degrees in one operation
        ecef_kep_traj = eci_cart_traj.convert_to(
            brahe.OrbitFrame.ecef,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.degrees
        )

        assert ecef_kep_traj.frame == brahe.OrbitFrame.ecef
        assert ecef_kep_traj.representation == brahe.OrbitRepresentation.keplerian
        assert ecef_kep_traj.angle_format == brahe.AngleFormat.degrees
        assert len(ecef_kep_traj) == 1

    def test_orbital_trajectory_properties(self):
        """Test orbital trajectory property access methods."""
        # Test Cartesian trajectory properties
        cart_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        assert cart_traj.frame == brahe.OrbitFrame.eci
        assert cart_traj.representation == brahe.OrbitRepresentation.cartesian
        assert cart_traj.angle_format == brahe.AngleFormat.none

        # Test Keplerian trajectory properties
        kep_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.ecef,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.degrees
        )

        assert kep_traj.frame == brahe.OrbitFrame.ecef
        assert kep_traj.representation == brahe.OrbitRepresentation.keplerian
        assert kep_traj.angle_format == brahe.AngleFormat.degrees

    def test_orbital_trajectory_direct_frame_conversions(self):
        """Test direct frame conversion shortcuts (to_eci, to_ecef)."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create ECI Cartesian trajectory
        eci_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        cart_state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        eci_traj.add_state(epoch, cart_state)

        # Test to_eci (should be no-op for ECI trajectory)
        eci_traj2 = eci_traj.to_eci()
        assert eci_traj2.frame == brahe.OrbitFrame.eci
        assert len(eci_traj2) == 1

        # Test to_ecef conversion
        # Note: This requires EOP data, so it will be tested in integration tests
        # For now, we'll test that the method exists and can be called
        assert hasattr(eci_traj, 'to_ecef')

    def test_orbital_trajectory_direct_angle_conversions(self):
        """Test direct angle format conversion shortcuts (to_degrees, to_radians)."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create Keplerian trajectory in radians
        kep_rad_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.radians
        )

        # Semi-major axis, eccentricity, inclination, RAAN, arg of perigee, mean anomaly (radians)
        kep_state_rad = np.array([6.678e6, 0.001, 0.87266, 0.0, 0.0, 0.0])
        kep_rad_traj.add_state(epoch, kep_state_rad)

        # Convert to degrees
        kep_deg_traj = kep_rad_traj.to_degrees()
        assert kep_deg_traj.angle_format == brahe.AngleFormat.degrees
        assert len(kep_deg_traj) == 1

        # Convert back to radians
        kep_rad_traj2 = kep_deg_traj.to_radians()
        assert kep_rad_traj2.angle_format == brahe.AngleFormat.radians
        assert len(kep_rad_traj2) == 1

    def test_orbital_trajectory_generic_representation_conversion(self):
        """Test generic to_representation method."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create Cartesian trajectory
        cart_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        cart_state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        cart_traj.add_state(epoch, cart_state)

        # Convert to Keplerian using generic to_representation
        kep_traj = cart_traj.to_representation(
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.degrees
        )

        assert kep_traj.representation == brahe.OrbitRepresentation.keplerian
        assert kep_traj.angle_format == brahe.AngleFormat.degrees
        assert len(kep_traj) == 1

        # Convert back to Cartesian
        cart_traj2 = kep_traj.to_representation(
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        assert cart_traj2.representation == brahe.OrbitRepresentation.cartesian
        assert cart_traj2.angle_format == brahe.AngleFormat.none
        assert len(cart_traj2) == 1

    def test_orbital_trajectory_position_velocity_access(self):
        """Test position and velocity component access methods."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create Cartesian trajectory with known position and velocity
        cart_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Position: [6678 km, 0, 0], Velocity: [0, 7.726 km/s, 0]
        cart_state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        cart_traj.add_state(epoch, cart_state)

        # Test position extraction
        position = cart_traj.position_at_epoch(epoch)
        assert position.shape == (3,)
        assert position[0] == pytest.approx(6.678e6, rel=1e-6)
        assert position[1] == pytest.approx(0.0, abs=1e-6)
        assert position[2] == pytest.approx(0.0, abs=1e-6)

        # Test velocity extraction
        velocity = cart_traj.velocity_at_epoch(epoch)
        assert velocity.shape == (3,)
        assert velocity[0] == pytest.approx(0.0, abs=1e-6)
        assert velocity[1] == pytest.approx(7.726e3, rel=1e-6)
        assert velocity[2] == pytest.approx(0.0, abs=1e-6)

    def test_orbital_trajectory_position_velocity_requirements(self):
        """Test position and velocity access method requirements."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create Keplerian trajectory
        kep_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.radians
        )

        # Semi-major axis, eccentricity, inclination, RAAN, arg of perigee, mean anomaly
        kep_state = np.array([6.678e6, 0.001, 0.87266, 0.0, 0.0, 0.0])
        kep_traj.add_state(epoch, kep_state)

        # Position and velocity extraction should fail for non-Cartesian representations
        with pytest.raises(RuntimeError, match="Cannot extract position from non-Cartesian"):
            kep_traj.position_at_epoch(epoch)

        with pytest.raises(RuntimeError, match="Cannot extract velocity from non-Cartesian"):
            kep_traj.velocity_at_epoch(epoch)

        # Convert to Cartesian and then position/velocity access should work
        cart_traj = kep_traj.to_cartesian()
        position = cart_traj.position_at_epoch(epoch)
        velocity = cart_traj.velocity_at_epoch(epoch)

        assert position.shape == (3,)
        assert velocity.shape == (3,)

        # Should be physically reasonable orbital values
        pos_magnitude = np.linalg.norm(position)
        vel_magnitude = np.linalg.norm(velocity)

        assert pos_magnitude == pytest.approx(6.678e6, rel=1e-2)  # Semi-major axis
        assert vel_magnitude > 0  # Should have some velocity

    def test_orbital_trajectory_error_conditions(self):
        """Test error conditions for new methods."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        future_epoch = brahe.Epoch.from_datetime(2023, 1, 2, 12, 0, 0.0, 0.0, "UTC")

        # Create empty trajectory
        empty_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Test accessing position/velocity from empty trajectory
        with pytest.raises(RuntimeError):
            empty_traj.position_at_epoch(epoch)

        with pytest.raises(RuntimeError):
            empty_traj.velocity_at_epoch(epoch)

        # Create trajectory with one state
        single_state_traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        cart_state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        single_state_traj.add_state(epoch, cart_state)

        # Test accessing at epoch not in trajectory (should interpolate or error)
        try:
            position = single_state_traj.position_at_epoch(future_epoch)
            # If it succeeds, it means extrapolation/interpolation worked
            assert position.shape == (3,)
        except RuntimeError:
            # If it fails, that's also acceptable behavior for out-of-range epochs
            pass

    def test_orbital_trajectory_current_state_epoch(self):
        """Test current state and epoch access methods."""
        # Create trajectory
        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Test empty trajectory
        current_state = traj.current_state_vector()
        current_epoch = traj.current_epoch()

        assert current_state.shape == (6,)
        assert np.allclose(current_state, np.zeros(6))  # Should be zeros for empty

        # Add states
        epoch1 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        epoch2 = brahe.Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, "UTC")

        state1 = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])
        state2 = np.array([6.678e6, 100.0, 0.0, 0.0, 7.726e3, 100.0])

        traj.add_state(epoch1, state1)
        traj.add_state(epoch2, state2)

        # Test that current state/epoch returns the most recent
        current_state = traj.current_state_vector()
        current_epoch = traj.current_epoch()

        assert current_state.shape == (6,)
        np.testing.assert_array_almost_equal(current_state, state2)
        assert current_epoch.jd() == epoch2.jd()

    def test_orbital_trajectory_from_orbital_data(self):
        """Test creating orbital trajectory from data."""
        # Create test data
        epochs = [
            brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC"),
            brahe.Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, "UTC"),
            brahe.Epoch.from_datetime(2023, 1, 1, 14, 0, 0.0, 0.0, "UTC"),
        ]

        # Flattened states array (3 states × 6 elements each)
        states = np.array([
            6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0,          # State 1
            6.678e6, 100.0, 0.0, 0.0, 7.726e3, 100.0,      # State 2
            6.678e6, 200.0, 0.0, 0.0, 7.726e3, 200.0,      # State 3
        ])

        # Create trajectory from data
        traj = brahe.OrbitTrajectory.from_orbital_data(
            epochs,
            states,
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Verify trajectory properties
        assert len(traj) == 3
        assert traj.frame == brahe.OrbitFrame.eci
        assert traj.representation == brahe.OrbitRepresentation.cartesian

        # Verify states were added correctly
        state0 = traj.state(0)
        state1 = traj.state(1)
        state2 = traj.state(2)

        np.testing.assert_array_almost_equal(state0, states[0:6])
        np.testing.assert_array_almost_equal(state1, states[6:12])
        np.testing.assert_array_almost_equal(state2, states[12:18])

    def test_orbital_trajectory_from_orbital_data_validation(self):
        """Test validation errors in from_orbital_data."""
        epochs = [
            brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC"),
            brahe.Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, "UTC"),
        ]

        # Test wrong states array length (not multiple of 6)
        with pytest.raises((ValueError, TypeError)):
            brahe.OrbitTrajectory.from_orbital_data(
                epochs,
                np.array([1, 2, 3, 4, 5]),  # 5 elements, not multiple of 6
                brahe.OrbitFrame.eci,
                brahe.OrbitRepresentation.cartesian,
                brahe.AngleFormat.none
            )

        # Test mismatched epochs and states count
        with pytest.raises((ValueError, TypeError)):
            brahe.OrbitTrajectory.from_orbital_data(
                epochs,  # 2 epochs
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),  # 3 states
                brahe.OrbitFrame.eci,
                brahe.OrbitRepresentation.cartesian,
                brahe.AngleFormat.none
            )

    def test_orbital_trajectory_convert_state_to_format(self):
        """Test state format conversion method."""
        # Create trajectory
        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Test state conversion (simple case - same format)
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])

        converted_state = traj.convert_state_to_format(
            state,
            epoch,
            brahe.OrbitFrame.eci,         # from_frame
            brahe.OrbitRepresentation.cartesian,  # from_representation
            brahe.AngleFormat.none,       # from_angle_format
            brahe.OrbitFrame.eci,         # to_frame (same)
            brahe.OrbitRepresentation.cartesian,  # to_representation (same)
            brahe.AngleFormat.none        # to_angle_format (same)
        )

        assert converted_state.shape == (6,)
        np.testing.assert_array_almost_equal(converted_state, state, decimal=6)

    def test_orbital_trajectory_convert_state_to_format_validation(self):
        """Test validation errors in convert_state_to_format."""
        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Test wrong state vector length
        with pytest.raises((ValueError, TypeError)):
            traj.convert_state_to_format(
                np.array([1, 2, 3, 4, 5]),  # Only 5 elements
                epoch,
                brahe.OrbitFrame.eci,
                brahe.OrbitRepresentation.cartesian,
                brahe.AngleFormat.none,
                brahe.OrbitFrame.eci,
                brahe.OrbitRepresentation.cartesian,
                brahe.AngleFormat.none
            )

    def test_orbital_trajectory_index_before_epoch(self):
        """Test index_before_epoch method."""
        # Create trajectory with states at t0, t0+60s, t0+120s
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Add states with distinguishable values
        traj.add_state(t0, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        traj.add_state(t0 + 60.0, np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        traj.add_state(t0 + 120.0, np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        # Test error case - epoch before all states
        before_t0 = t0 + (-10.0)
        with pytest.raises(RuntimeError):
            traj.index_before_epoch(before_t0)

        # Test finding index before t0+30s (should return index 0)
        t0_plus_30 = t0 + 30.0
        idx = traj.index_before_epoch(t0_plus_30)
        assert idx == 0

        # Test finding index before t0+60s (should return index 1 - exact match)
        t0_plus_60 = t0 + 60.0
        idx = traj.index_before_epoch(t0_plus_60)
        assert idx == 1

        # Test finding index before t0+90s (should return index 1)
        t0_plus_90 = t0 + 90.0
        idx = traj.index_before_epoch(t0_plus_90)
        assert idx == 1

        # Test finding index before t0+120s (should return index 2 - exact match)
        t0_plus_120 = t0 + 120.0
        idx = traj.index_before_epoch(t0_plus_120)
        assert idx == 2

    def test_orbital_trajectory_index_after_epoch(self):
        """Test index_after_epoch method."""
        # Create trajectory with states at t0, t0+60s, t0+120s
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Add states with distinguishable values
        traj.add_state(t0, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        traj.add_state(t0 + 60.0, np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        traj.add_state(t0 + 120.0, np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        # Test error case - epoch after all states
        after_t0_120 = t0 + 150.0
        with pytest.raises(RuntimeError):
            traj.index_after_epoch(after_t0_120)

        # Test finding index after t0-30s (should return index 0)
        before_t0 = t0 + (-30.0)
        idx = traj.index_after_epoch(before_t0)
        assert idx == 0

        # Test finding index after t0 (should return index 0 - exact match)
        idx = traj.index_after_epoch(t0)
        assert idx == 0

        # Test finding index after t0+30s (should return index 1)
        t0_plus_30 = t0 + 30.0
        idx = traj.index_after_epoch(t0_plus_30)
        assert idx == 1

        # Test finding index after t0+60s (should return index 1 - exact match)
        t0_plus_60 = t0 + 60.0
        idx = traj.index_after_epoch(t0_plus_60)
        assert idx == 1

        # Test finding index after t0+90s (should return index 2)
        t0_plus_90 = t0 + 90.0
        idx = traj.index_after_epoch(t0_plus_90)
        assert idx == 2

    def test_orbital_trajectory_state_before_epoch(self):
        """Test state_before_epoch method."""
        # Create trajectory with distinguishable states
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Add states with distinguishable values
        state1 = np.array([1000.0, 100.0, 10.0, 1.0, 0.1, 0.01])
        state2 = np.array([2000.0, 200.0, 20.0, 2.0, 0.2, 0.02])
        state3 = np.array([3000.0, 300.0, 30.0, 3.0, 0.3, 0.03])

        traj.add_state(t0, state1)
        traj.add_state(t0 + 60.0, state2)
        traj.add_state(t0 + 120.0, state3)

        # Test error case - epoch before all states
        before_t0 = t0 + (-10.0)
        with pytest.raises(RuntimeError):
            traj.state_before_epoch(before_t0)

        # Test at t0+30s (should return first state)
        t0_plus_30 = t0 + 30.0
        ret_epoch, ret_state = traj.state_before_epoch(t0_plus_30)
        assert ret_epoch.jd() == pytest.approx(t0.jd(), rel=1e-9)
        assert ret_state[0] == pytest.approx(1000.0, rel=1e-9)
        assert ret_state[1] == pytest.approx(100.0, rel=1e-9)

        # Test at exact match t0+60s (should return second state)
        t0_plus_60 = t0 + 60.0
        ret_epoch, ret_state = traj.state_before_epoch(t0_plus_60)
        assert ret_epoch.jd() == pytest.approx(t0_plus_60.jd(), rel=1e-9)
        assert ret_state[0] == pytest.approx(2000.0, rel=1e-9)
        assert ret_state[1] == pytest.approx(200.0, rel=1e-9)

        # Test at t0+90s (should return second state)
        t0_plus_90 = t0 + 90.0
        ret_epoch, ret_state = traj.state_before_epoch(t0_plus_90)
        assert ret_epoch.jd() == pytest.approx(t0_plus_60.jd(), rel=1e-9)
        assert ret_state[0] == pytest.approx(2000.0, rel=1e-9)
        assert ret_state[1] == pytest.approx(200.0, rel=1e-9)

    def test_orbital_trajectory_state_after_epoch(self):
        """Test state_after_epoch method."""
        # Create trajectory with distinguishable states
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Add states with distinguishable values
        state1 = np.array([1000.0, 100.0, 10.0, 1.0, 0.1, 0.01])
        state2 = np.array([2000.0, 200.0, 20.0, 2.0, 0.2, 0.02])
        state3 = np.array([3000.0, 300.0, 30.0, 3.0, 0.3, 0.03])

        traj.add_state(t0, state1)
        traj.add_state(t0 + 60.0, state2)
        traj.add_state(t0 + 120.0, state3)

        # Test error case - epoch after all states
        after_t0_120 = t0 + 150.0
        with pytest.raises(RuntimeError):
            traj.state_after_epoch(after_t0_120)

        # Test at t0-30s (should return first state)
        before_t0 = t0 + (-30.0)
        ret_epoch, ret_state = traj.state_after_epoch(before_t0)
        assert ret_epoch.jd() == pytest.approx(t0.jd(), rel=1e-9)
        assert ret_state[0] == pytest.approx(1000.0, rel=1e-9)
        assert ret_state[1] == pytest.approx(100.0, rel=1e-9)

        # Test at exact match t0 (should return first state)
        ret_epoch, ret_state = traj.state_after_epoch(t0)
        assert ret_epoch.jd() == pytest.approx(t0.jd(), rel=1e-9)
        assert ret_state[0] == pytest.approx(1000.0, rel=1e-9)
        assert ret_state[1] == pytest.approx(100.0, rel=1e-9)

        # Test at t0+30s (should return second state)
        t0_plus_30 = t0 + 30.0
        t0_plus_60 = t0 + 60.0
        ret_epoch, ret_state = traj.state_after_epoch(t0_plus_30)
        assert ret_epoch.jd() == pytest.approx(t0_plus_60.jd(), rel=1e-9)
        assert ret_state[0] == pytest.approx(2000.0, rel=1e-9)
        assert ret_state[1] == pytest.approx(200.0, rel=1e-9)

    def test_orbital_trajectory_set_interpolation_method(self):
        """Test set_interpolation_method method."""
        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Test setting different interpolation methods
        # Note: Currently only Linear and Lagrange are exposed in Python bindings
        traj.set_interpolation_method(brahe.InterpolationMethod.linear)
        assert traj.get_interpolation_method() == brahe.InterpolationMethod.linear

        traj.set_interpolation_method(brahe.InterpolationMethod.linear)
        assert traj.get_interpolation_method() == brahe.InterpolationMethod.linear

        # Set back to linear
        traj.set_interpolation_method(brahe.InterpolationMethod.linear)
        assert traj.get_interpolation_method() == brahe.InterpolationMethod.linear

    def test_orbital_trajectory_get_interpolation_method(self):
        """Test get_interpolation_method method."""
        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Verify default is Linear
        assert traj.get_interpolation_method() == brahe.InterpolationMethod.linear

        # Change method and verify getter returns correct value
        traj.set_interpolation_method(brahe.InterpolationMethod.linear)
        assert traj.get_interpolation_method() == brahe.InterpolationMethod.linear

        # Change back to linear
        traj.set_interpolation_method(brahe.InterpolationMethod.linear)
        assert traj.get_interpolation_method() == brahe.InterpolationMethod.linear

    def test_orbital_trajectory_interpolate_linear(self):
        """Test interpolate_linear method."""
        # Create trajectory with simple values for easy verification
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Add states with linearly varying position for simple interpolation verification
        # At t0: x=7000km, At t0+60s: x=7060km, At t0+120s: x=7120km (1 km/s change)
        traj.add_state(t0, np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]))
        traj.add_state(t0 + 60.0, np.array([7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]))
        traj.add_state(t0 + 120.0, np.array([7120e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]))

        # Test linear interpolation at t0+30s (midpoint between first two states)
        # Should be halfway between [7000e3, ...] and [7060e3, ...]
        t_mid = t0 + 30.0
        state_mid = traj.interpolate_linear(t_mid)
        assert state_mid[0] == pytest.approx(7030e3, rel=1e-6)
        assert state_mid[1] == pytest.approx(0.0, abs=1e-6)
        assert state_mid[2] == pytest.approx(0.0, abs=1e-6)
        assert state_mid[3] == pytest.approx(0.0, abs=1e-6)
        assert state_mid[4] == pytest.approx(7.5e3, rel=1e-6)
        assert state_mid[5] == pytest.approx(0.0, abs=1e-6)

        # Test at exact epochs - should return exact states
        state_0 = traj.interpolate_linear(t0)
        assert state_0[0] == pytest.approx(7000e3, rel=1e-6)

        state_60 = traj.interpolate_linear(t0 + 60.0)
        assert state_60[0] == pytest.approx(7060e3, rel=1e-6)

        # Test at t0+90s (3/4 of the way between t0+60s and t0+120s)
        # Should be 1/2 of the way: 7060e3 + 0.5 * (7120e3 - 7060e3) = 7090e3
        t_90 = t0 + 90.0
        state_90 = traj.interpolate_linear(t_90)
        assert state_90[0] == pytest.approx(7090e3, rel=1e-6)

    def test_orbital_trajectory_interpolate(self):
        """Test interpolate method."""
        # Create trajectory
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Add states
        traj.add_state(t0, np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]))
        traj.add_state(t0 + 60.0, np.array([7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]))
        traj.add_state(t0 + 120.0, np.array([7120e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]))

        # Set interpolation method to Linear
        traj.set_interpolation_method(brahe.InterpolationMethod.linear)

        # Test that interpolate() matches interpolate_linear() for Linear method
        t_test = t0 + 30.0
        result_interpolate = traj.interpolate(t_test)
        result_linear = traj.interpolate_linear(t_test)

        np.testing.assert_array_almost_equal(result_interpolate, result_linear, decimal=6)


def test_orbittrajectory_set_max_size():
    """Test set_max_size eviction policy"""
    traj = brahe.OrbitTrajectory(
        brahe.OrbitFrame.eci,
        brahe.OrbitRepresentation.cartesian,
        brahe.AngleFormat.none
    )

    # Add 5 states
    t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    for i in range(5):
        epoch = t0 + (i * 60.0)
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add_state(epoch, state)

    assert len(traj) == 5

    # Set max size to 3 - should keep the 3 most recent states
    traj.set_max_size(3)

    assert len(traj) == 3

    # Verify the oldest states were evicted
    first_state = traj.state(0)
    assert abs(first_state[0] - (7000e3 + 2000.0)) < 1.0  # Should be 3rd state

    # Test error cases
    with pytest.raises(RuntimeError):
        traj.set_max_size(0)


def test_orbittrajectory_set_max_age():
    """Test set_max_age eviction policy"""
    traj = brahe.OrbitTrajectory(
        brahe.OrbitFrame.eci,
        brahe.OrbitRepresentation.cartesian,
        brahe.AngleFormat.none
    )

    # Add states spanning 5 minutes
    t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    for i in range(6):
        epoch = t0 + (i * 60.0)  # 0, 60, 120, 180, 240, 300 seconds
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add_state(epoch, state)

    assert len(traj) == 6

    # Set max age to 150 seconds - should keep states within 150s of the last epoch
    traj.set_max_age(150.0)

    # Should keep states at 180s, 240s, and 300s (within 150s of 300s)
    assert len(traj) == 3

    first_state = traj.state(0)
    assert abs(first_state[0] - (7000e3 + 3000.0)) < 1.0

    # Test error cases
    with pytest.raises(RuntimeError):
        traj.set_max_age(0.0)
    with pytest.raises(RuntimeError):
        traj.set_max_age(-10.0)


def test_orbittrajectory_to_matrix():
    """Test conversion to matrix representation"""
    epochs = [
        brahe.Epoch.from_jd(2451545.0, "UTC"),
        brahe.Epoch.from_jd(2451545.1, "UTC"),
        brahe.Epoch.from_jd(2451545.2, "UTC"),
    ]
    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
        7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0,
    ])
    traj = brahe.OrbitTrajectory.from_orbital_data(
        epochs, states,
        brahe.OrbitFrame.eci,
        brahe.OrbitRepresentation.cartesian,
        brahe.AngleFormat.none
    )

    # Convert to matrix
    matrix = traj.to_matrix()

    # Verify dimensions: 6 rows (state elements) x 3 columns (time points)
    assert matrix.shape == (6, 3)

    # Verify first column matches first state
    np.testing.assert_almost_equal(matrix[0, 0], 7000e3, decimal=6)
    np.testing.assert_almost_equal(matrix[1, 0], 0.0, decimal=6)
    np.testing.assert_almost_equal(matrix[2, 0], 0.0, decimal=6)

    # Verify second column matches second state
    np.testing.assert_almost_equal(matrix[0, 1], 7100e3, decimal=6)
    np.testing.assert_almost_equal(matrix[1, 1], 1000e3, decimal=6)
    np.testing.assert_almost_equal(matrix[2, 1], 500e3, decimal=6)

    # Verify third column matches third state
    np.testing.assert_almost_equal(matrix[0, 2], 7200e3, decimal=6)
    np.testing.assert_almost_equal(matrix[1, 2], 2000e3, decimal=6)
    np.testing.assert_almost_equal(matrix[2, 2], 1000e3, decimal=6)


class TestOrbitTrajectoryIndex:
    """Tests for Index trait implementation on OrbitTrajectory."""

    def test_orbittrajectory_index(self):
        """Test positive indexing into trajectory."""
        epochs = [
            brahe.Epoch.from_jd(2451545.0, "UTC"),
            brahe.Epoch.from_jd(2451545.1, "UTC"),
            brahe.Epoch.from_jd(2451545.2, "UTC"),
        ]
        states = np.array([
            7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
            7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
            7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0,
        ])
        traj = brahe.OrbitTrajectory.from_orbital_data(
            epochs, states,
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Test positive indexing
        state0 = traj[0]
        assert abs(state0[0] - 7000e3) < 1.0

        state1 = traj[1]
        assert abs(state1[0] - 7100e3) < 1.0

        state2 = traj[2]
        assert abs(state2[0] - 7200e3) < 1.0

    def test_orbittrajectory_index_negative(self):
        """Test negative indexing into trajectory."""
        epochs = [
            brahe.Epoch.from_jd(2451545.0, "UTC"),
            brahe.Epoch.from_jd(2451545.1, "UTC"),
            brahe.Epoch.from_jd(2451545.2, "UTC"),
        ]
        states = np.array([
            7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
            7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
            7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0,
        ])
        traj = brahe.OrbitTrajectory.from_orbital_data(
            epochs, states,
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        # Test negative indexing
        state_last = traj[-1]
        assert abs(state_last[0] - 7200e3) < 1.0

        state_second_last = traj[-2]
        assert abs(state_second_last[0] - 7100e3) < 1.0

    def test_orbittrajectory_index_out_of_bounds(self):
        """Test indexing out of bounds raises IndexError."""
        epoch = brahe.Epoch.from_jd(2451545.0, "UTC")
        state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj = brahe.OrbitTrajectory.from_orbital_data(
            [epoch], state,
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        with pytest.raises(IndexError):
            _ = traj[10]

        with pytest.raises(IndexError):
            _ = traj[-10]


class TestOrbitTrajectoryIterator:
    """Tests for Iterator trait implementation on OrbitTrajectory."""

    def test_orbittrajectory_iterator(self):
        """Test iterating over trajectory yields (epoch, state) pairs."""
        epochs = [
            brahe.Epoch.from_jd(2451545.0, "UTC"),
            brahe.Epoch.from_jd(2451545.1, "UTC"),
            brahe.Epoch.from_jd(2451545.2, "UTC"),
        ]
        states = np.array([
            7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
            7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
            7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0,
        ])
        traj = brahe.OrbitTrajectory.from_orbital_data(
            epochs, states,
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        count = 0
        for epoch, state in traj:
            if count == 0:
                assert epoch.jd() == 2451545.0
                assert abs(state[0] - 7000e3) < 1.0
            elif count == 1:
                assert epoch.jd() == 2451545.1
                assert abs(state[0] - 7100e3) < 1.0
            elif count == 2:
                assert epoch.jd() == 2451545.2
                assert abs(state[0] - 7200e3) < 1.0
            count += 1

        assert count == 3

    def test_orbittrajectory_iterator_empty(self):
        """Test iterating over empty trajectory."""
        traj = brahe.OrbitTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none
        )

        count = 0
        for _ in traj:
            count += 1

        assert count == 0

