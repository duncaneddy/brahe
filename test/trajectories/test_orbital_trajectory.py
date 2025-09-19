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

    def test_orbital_trajectory_representation_conversion(self):
        """Test representation conversion between Cartesian and Keplerian."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create Cartesian trajectory
        cart_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
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
        eci_cart_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
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
        cart_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        assert cart_traj.frame == brahe.OrbitFrame.eci
        assert cart_traj.representation == brahe.OrbitRepresentation.cartesian
        assert cart_traj.angle_format == brahe.AngleFormat.none

        # Test Keplerian trajectory properties
        kep_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.ecef,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.degrees,
            brahe.InterpolationMethod.linear
        )

        assert kep_traj.frame == brahe.OrbitFrame.ecef
        assert kep_traj.representation == brahe.OrbitRepresentation.keplerian
        assert kep_traj.angle_format == brahe.AngleFormat.degrees

    def test_orbital_trajectory_direct_frame_conversions(self):
        """Test direct frame conversion shortcuts (to_eci, to_ecef)."""
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Create ECI Cartesian trajectory
        eci_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
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
        kep_rad_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.radians,
            brahe.InterpolationMethod.linear
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
        cart_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
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
        cart_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
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
        kep_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.keplerian,
            brahe.AngleFormat.radians,
            brahe.InterpolationMethod.linear
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
        empty_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # Test accessing position/velocity from empty trajectory
        with pytest.raises(RuntimeError):
            empty_traj.position_at_epoch(epoch)

        with pytest.raises(RuntimeError):
            empty_traj.velocity_at_epoch(epoch)

        # Create trajectory with one state
        single_state_traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
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
        traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
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
        traj = brahe.OrbitalTrajectory.from_orbital_data(
            epochs,
            states,
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
        )

        # Verify trajectory properties
        assert len(traj) == 3
        assert traj.frame == brahe.OrbitFrame.eci
        assert traj.representation == brahe.OrbitRepresentation.cartesian

        # Verify states were added correctly
        state0 = traj.state_at_index(0)
        state1 = traj.state_at_index(1)
        state2 = traj.state_at_index(2)

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
            brahe.OrbitalTrajectory.from_orbital_data(
                epochs,
                np.array([1, 2, 3, 4, 5]),  # 5 elements, not multiple of 6
                brahe.OrbitFrame.eci,
                brahe.OrbitRepresentation.cartesian,
                brahe.AngleFormat.none,
                brahe.InterpolationMethod.linear
            )

        # Test mismatched epochs and states count
        with pytest.raises((ValueError, TypeError)):
            brahe.OrbitalTrajectory.from_orbital_data(
                epochs,  # 2 epochs
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),  # 3 states
                brahe.OrbitFrame.eci,
                brahe.OrbitRepresentation.cartesian,
                brahe.AngleFormat.none,
                brahe.InterpolationMethod.linear
            )

    def test_orbital_trajectory_convert_state_to_format(self):
        """Test state format conversion method."""
        # Create trajectory
        traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
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
        traj = brahe.OrbitalTrajectory(
            brahe.OrbitFrame.eci,
            brahe.OrbitRepresentation.cartesian,
            brahe.AngleFormat.none,
            brahe.InterpolationMethod.linear
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