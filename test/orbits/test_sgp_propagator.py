"""
Tests for SGP4 propagator functionality in brahe.

The SGP4 propagator implements the Simplified General Perturbations 4 model
for satellite orbit prediction using Two-Line Element (TLE) data.
"""

import pytest
import numpy as np
import brahe


@pytest.fixture
def iss_classic_tle():
    """ISS TLE in classic format for testing."""
    return (
        "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992",
        "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
    )


@pytest.fixture
def iss_3le():
    """ISS TLE in 3-line format with satellite name."""
    return (
        "ISS (ZARYA)",
        "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992",
        "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
    )


@pytest.fixture
def alpha5_tle():
    """TLE with alpha-5 NORAD ID format for testing."""
    return (
        "1 A0001U 23001A   23001.00000000  .00000000  00000-0  00000-0 0  9994",
        "2 A0001  00.0000 000.0000 0000000  00.0000 000.0000 01.00000000000009"
    )


@pytest.fixture
def sgp_propagator(iss_classic_tle):
    """SGP propagator created from ISS TLE."""
    return brahe.SGPPropagator.from_tle(iss_classic_tle[0], iss_classic_tle[1])


class TestSGPPropagatorCreation:
    """Test SGP propagator creation from different TLE formats."""

    def test_from_tle_basic(self, iss_classic_tle):
        """Test SGP propagator creation from 2-line TLE format."""
        sgp = brahe.SGPPropagator.from_tle(iss_classic_tle[0], iss_classic_tle[1])

        assert sgp.norad_id == 25544
        assert sgp.satellite_name is None
        assert sgp.step_size == 60.0  # Default step size

    def test_from_tle_with_step_size(self, iss_classic_tle):
        """Test SGP propagator creation with custom step size."""
        sgp = brahe.SGPPropagator.from_tle(iss_classic_tle[0], iss_classic_tle[1], step_size=120.0)

        assert sgp.norad_id == 25544
        assert sgp.step_size == 120.0

    def test_from_3le_basic(self, iss_3le):
        """Test SGP propagator creation from 3-line TLE format."""
        sgp = brahe.SGPPropagator.from_3le(iss_3le[0], iss_3le[1], iss_3le[2])

        assert sgp.norad_id == 25544
        assert sgp.satellite_name == "ISS (ZARYA)"
        assert sgp.step_size == 60.0  # Default step size

    def test_from_3le_with_step_size(self, iss_3le):
        """Test SGP propagator creation from 3-line TLE with custom step size."""
        sgp = brahe.SGPPropagator.from_3le(iss_3le[0], iss_3le[1], iss_3le[2], step_size=30.0)

        assert sgp.norad_id == 25544
        assert sgp.satellite_name == "ISS (ZARYA)"
        assert sgp.step_size == 30.0

    def test_alpha5_norad_id(self, alpha5_tle):
        """Test SGP propagator creation with alpha-5 NORAD ID format."""
        sgp = brahe.SGPPropagator.from_tle(alpha5_tle[0], alpha5_tle[1])

        # Alpha-5 "A0001" should decode to 100001
        expected_id = 10 * 10000 + 1  # A=10, 0001=1
        assert sgp.norad_id == expected_id

    def test_invalid_tle_lines(self):
        """Test error handling for invalid TLE lines."""
        with pytest.raises(RuntimeError):
            brahe.SGPPropagator.from_tle("invalid line 1", "invalid line 2")

    def test_mismatched_norad_ids(self):
        """Test error handling for mismatched NORAD IDs in TLE lines."""
        line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992"
        line2 = "2 12345  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"

        with pytest.raises(RuntimeError):
            brahe.SGPPropagator.from_tle(line1, line2)


class TestSGPPropagatorConfiguration:
    """Test SGP propagator configuration and settings."""

    def test_step_size_modification(self, sgp_propagator):
        """Test modification of step size."""
        assert sgp_propagator.step_size == 60.0

        sgp_propagator.step_size = 120.0
        assert sgp_propagator.step_size == 120.0

    def test_output_format_cartesian(self, sgp_propagator):
        """Test setting output format to Cartesian."""
        sgp_propagator.set_output_cartesian()

        # Test that state computation works
        state = sgp_propagator.state(sgp_propagator.epoch)
        assert len(state) == 6  # Position and velocity components

    def test_output_format_keplerian(self, sgp_propagator):
        """Test setting output format to Keplerian elements."""
        sgp_propagator.set_output_keplerian()

        # Test that state computation works
        state = sgp_propagator.state(sgp_propagator.epoch)
        assert len(state) == 6  # Orbital elements

    def test_output_frame_settings(self, sgp_propagator):
        """Test setting different output frames."""
        # Test ECI frame
        sgp_propagator.set_output_frame(brahe.OrbitFrame.eci)
        state_eci = sgp_propagator.state(sgp_propagator.epoch)

        # Test ECEF frame
        sgp_propagator.set_output_frame(brahe.OrbitFrame.ecef)
        state_ecef = sgp_propagator.state(sgp_propagator.epoch)

        # States should be different due to frame transformation
        assert not np.allclose(state_eci, state_ecef)

    def test_output_angle_format(self, sgp_propagator):
        """Test setting different angle formats for Keplerian output."""
        sgp_propagator.set_output_keplerian()

        # Test radians
        sgp_propagator.set_output_angle_format(brahe.AngleFormat.radians)
        state_rad = sgp_propagator.state(sgp_propagator.epoch)

        # Test degrees
        sgp_propagator.set_output_angle_format(brahe.AngleFormat.degrees)
        state_deg = sgp_propagator.state(sgp_propagator.epoch)

        # Angular elements should be different
        # Semi-major axis and eccentricity should be the same
        assert np.isclose(state_rad[0], state_deg[0])  # a
        assert np.isclose(state_rad[1], state_deg[1])  # e
        # Angular elements should differ by conversion factor
        assert not np.isclose(state_rad[2], state_deg[2])  # i


class TestSGPPropagatorStates:
    """Test SGP propagator state computation."""

    def test_state_at_epoch(self, sgp_propagator):
        """Test state computation at the TLE epoch."""
        epoch = sgp_propagator.epoch
        state = sgp_propagator.state(epoch)

        assert len(state) == 6
        # State should be finite and reasonable for LEO
        assert np.all(np.isfinite(state))

        # Position magnitude should be reasonable for ISS (6000-7000 km)
        pos_mag = np.linalg.norm(state[:3])
        assert 6000e3 < pos_mag < 8000e3  # Convert km to m

    def test_state_eci(self, sgp_propagator):
        """Test ECI state computation."""
        epoch = sgp_propagator.epoch
        state = sgp_propagator.state_eci(epoch)

        assert len(state) == 6
        assert np.all(np.isfinite(state))

    def test_state_ecef(self, sgp_propagator):
        """Test ECEF state computation."""
        epoch = sgp_propagator.epoch
        state = sgp_propagator.state_ecef(epoch)

        assert len(state) == 6
        assert np.all(np.isfinite(state))

    def test_multiple_epochs(self, sgp_propagator):
        """Test state computation at multiple epochs."""
        base_epoch = sgp_propagator.epoch

        # Create epochs at different times
        epochs = [
            base_epoch,
            brahe.Epoch.from_jd(base_epoch.jd() + 1.0, brahe.TimeSystem.UTC),  # +1 day
            brahe.Epoch.from_jd(base_epoch.jd() - 1.0, brahe.TimeSystem.UTC),  # -1 day
        ]

        trajectory = sgp_propagator.states(epochs)
        assert trajectory.length == 3

    def test_states_eci(self, sgp_propagator):
        """Test ECI state computation for multiple epochs."""
        base_epoch = sgp_propagator.epoch
        epochs = [
            base_epoch,
            brahe.Epoch.from_jd(base_epoch.jd() + 0.1, brahe.TimeSystem.UTC),
        ]

        trajectory = sgp_propagator.states_eci(epochs)
        assert trajectory.length == 2
        assert trajectory.frame.name() == "Earth-Centered Inertial (J2000)"


class TestSGPPropagatorPropagation:
    """Test SGP propagator propagation methods."""

    def test_step_forward(self, sgp_propagator):
        """Test single step forward propagation."""
        initial_epoch = sgp_propagator.current_epoch

        sgp_propagator.step()

        new_epoch = sgp_propagator.current_epoch
        time_diff = new_epoch.jd() - initial_epoch.jd()
        expected_diff = sgp_propagator.step_size / 86400.0  # Convert seconds to days

        assert np.isclose(time_diff, expected_diff, rtol=1e-10)

    def test_step_by_duration(self, sgp_propagator):
        """Test step by specific duration."""
        initial_epoch = sgp_propagator.current_epoch
        step_duration = 300.0  # 5 minutes

        sgp_propagator.step_by(step_duration)

        new_epoch = sgp_propagator.current_epoch
        time_diff = new_epoch.jd() - initial_epoch.jd()
        expected_diff = step_duration / 86400.0

        assert np.isclose(time_diff, expected_diff, rtol=1e-10)

    def test_propagate_to_epoch(self, sgp_propagator):
        """Test propagation to specific target epoch."""
        target_epoch = brahe.Epoch.from_jd(
            sgp_propagator.current_epoch.jd() + 1.0,  # +1 day
            brahe.TimeSystem.UTC
        )

        sgp_propagator.propagate_to(target_epoch)

        # Should be very close to target epoch
        time_diff = abs(sgp_propagator.current_epoch.jd() - target_epoch.jd())
        assert time_diff < 1e-10  # Very small difference

    def test_reset_propagator(self, sgp_propagator):
        """Test resetting propagator to initial conditions."""
        initial_epoch = sgp_propagator.current_epoch

        # Propagate forward
        sgp_propagator.step()
        sgp_propagator.step()
        assert sgp_propagator.current_epoch.jd() > initial_epoch.jd()

        # Reset and verify
        sgp_propagator.reset()
        assert np.isclose(sgp_propagator.current_epoch.jd(), initial_epoch.jd())

    def test_trajectory_accumulation(self, sgp_propagator):
        """Test that propagator accumulates trajectory data."""
        initial_length = sgp_propagator.trajectory.length

        sgp_propagator.step()
        sgp_propagator.step()

        final_length = sgp_propagator.trajectory.length
        assert final_length > initial_length


class TestSGPPropagatorTLEUtilities:
    """Test TLE utility functions."""

    def test_validate_tle_lines_valid(self, iss_classic_tle):
        """Test TLE line validation with valid lines."""
        assert brahe.validate_tle_lines(iss_classic_tle[0], iss_classic_tle[1])

    def test_validate_tle_lines_invalid(self):
        """Test TLE line validation with invalid lines."""
        assert not brahe.validate_tle_lines("invalid line 1", "invalid line 2")

    def test_validate_tle_line_valid(self):
        """Test single TLE line validation."""
        valid_line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992"
        assert brahe.validate_tle_line(valid_line)

    def test_validate_tle_line_invalid(self):
        """Test single TLE line validation with invalid line."""
        assert not brahe.validate_tle_line("invalid line")

    def test_calculate_tle_line_checksum(self):
        """Test TLE line checksum calculation."""
        line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  999"
        checksum = brahe.calculate_tle_line_checksum(line)

        # Calculate expected checksum manually for verification
        # Count digits and minus signs in first 68 characters
        expected = 2  # Should be calculated based on the actual line content
        assert checksum == expected

    def test_extract_tle_norad_id_classic(self):
        """Test NORAD ID extraction from classic format."""
        norad_id = brahe.extract_tle_norad_id("25544")
        assert norad_id == 25544

    def test_extract_tle_norad_id_alpha5(self):
        """Test NORAD ID extraction from alpha-5 format."""
        # "A0001" should decode to 100001
        norad_id = brahe.extract_tle_norad_id("A0001")
        expected = 10 * 10000 + 1  # A=10, 0001=1
        assert norad_id == expected

    def test_extract_epoch_placeholder(self, iss_classic_tle):
        """Test epoch extraction (placeholder implementation)."""
        # Current implementation returns a placeholder epoch
        epoch = brahe.extract_epoch(iss_classic_tle[0])
        assert isinstance(epoch, brahe.Epoch)


class TestSGPPropagatorLegacyCompatibility:
    """Test legacy TLE compatibility functions."""

    def test_lines_to_orbit_elements(self, iss_classic_tle):
        """Test conversion of TLE lines to orbital elements."""
        elements = brahe.lines_to_orbit_elements(iss_classic_tle[0], iss_classic_tle[1])

        assert len(elements) == 6
        assert np.all(np.isfinite(elements))

    def test_lines_to_orbit_state(self, iss_classic_tle):
        """Test conversion of TLE lines to orbit state."""
        state = brahe.lines_to_orbit_state(iss_classic_tle[0], iss_classic_tle[1])

        assert len(state) == 6
        assert np.all(np.isfinite(state))


class TestSGPPropagatorErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_step_size(self, iss_classic_tle):
        """Test error handling for invalid step sizes."""
        sgp = brahe.SGPPropagator.from_tle(iss_classic_tle[0], iss_classic_tle[1])

        # Negative step size should work (backward propagation)
        sgp.step_size = -60.0
        assert sgp.step_size == -60.0

    def test_extreme_epoch_propagation(self, sgp_propagator):
        """Test propagation to extreme epochs."""
        # Test propagation far into the future (SGP4 may become inaccurate)
        far_future = brahe.Epoch.from_jd(
            sgp_propagator.epoch.jd() + 365.0,  # +1 year
            brahe.TimeSystem.UTC
        )

        # Should not crash, though accuracy may be poor
        sgp_propagator.propagate_to(far_future)
        state = sgp_propagator.state(far_future)
        assert len(state) == 6

    def test_string_representation(self, sgp_propagator):
        """Test string representation of SGP propagator."""
        repr_str = repr(sgp_propagator)
        assert "SGPPropagator" in repr_str
        assert "25544" in repr_str  # NORAD ID

        str_str = str(sgp_propagator)
        assert "SGPPropagator" in str_str


class TestSGPPropagatorNumericalAccuracy:
    """Test numerical accuracy and consistency of SGP propagator."""

    def test_propagation_consistency(self, sgp_propagator):
        """Test that forward and backward propagation are consistent."""
        initial_epoch = sgp_propagator.current_epoch
        initial_state = sgp_propagator.state(initial_epoch)

        # Propagate forward then backward
        sgp_propagator.step_by(3600.0)  # 1 hour forward
        sgp_propagator.step_by(-3600.0)  # 1 hour backward

        final_state = sgp_propagator.state(sgp_propagator.current_epoch)

        # Should return to approximately the same state
        assert np.allclose(initial_state, final_state, rtol=1e-10)

    def test_multiple_computation_consistency(self, sgp_propagator):
        """Test that multiple computations at same epoch give same result."""
        epoch = sgp_propagator.epoch

        state1 = sgp_propagator.state(epoch)
        state2 = sgp_propagator.state(epoch)

        assert np.allclose(state1, state2)

    def test_frame_transformation_consistency(self, sgp_propagator):
        """Test consistency between different frame computation methods."""
        epoch = sgp_propagator.epoch

        # Get state using frame-specific methods
        state_eci = sgp_propagator.state_eci(epoch)
        state_ecef = sgp_propagator.state_ecef(epoch)

        # Get state using general method with frame setting
        sgp_propagator.set_output_frame(brahe.OrbitFrame.eci)
        state_eci_general = sgp_propagator.state(epoch)

        sgp_propagator.set_output_frame(brahe.OrbitFrame.ecef)
        state_ecef_general = sgp_propagator.state(epoch)

        # Results should be consistent
        assert np.allclose(state_eci, state_eci_general)
        assert np.allclose(state_ecef, state_ecef_general)