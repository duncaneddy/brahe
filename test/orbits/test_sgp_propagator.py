"""
Tests for SGP4 propagator functionality in brahe.

These tests mirror the Rust test suite structure to ensure Python bindings work correctly.
"""

import pytest
import numpy as np
import brahe


# Test TLE data constants
ISS_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


@pytest.fixture
def iss_tle():
    """ISS TLE for testing."""
    return (ISS_LINE1, ISS_LINE2)


class TestSGPPropagatorMethods:
    """Test SGPPropagator struct methods."""

    def test_sgppropagator_from_tle(self, iss_tle):
        """Test SGPPropagator creation from TLE lines."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        assert prop.norad_id == 25544
        assert prop.step_size == 60.0
        assert prop.epoch.year() == 2008

    def test_sgppropagator_from_3le(self):
        """Test SGPPropagator creation from 3-line TLE."""
        line0 = "ISS (ZARYA)"
        line1 = ISS_LINE1
        line2 = ISS_LINE2

        prop = brahe.SGPPropagator.from_3le(line0, line1, line2, 60.0)

        assert prop.norad_id == 25544
        assert prop.step_size == 60.0
        assert prop.satellite_name == "ISS (ZARYA)"

    def test_sgppropagator_set_output_format_cartesian(self, iss_tle):
        """Test setting output format to ECI Cartesian."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_format(brahe.OrbitFrame.ECI, brahe.OrbitRepresentation.CARTESIAN, None)
        prop.step()

        # Verify it doesn't error and trajectory stores states
        assert prop.trajectory.length > 0

    def test_sgppropagator_set_output_format_keplerian(self, iss_tle):
        """Test setting output format to ECI Keplerian."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_format(brahe.OrbitFrame.ECI, brahe.OrbitRepresentation.KEPLERIAN, brahe.AngleFormat.RADIANS)
        prop.step()

        # Verify it doesn't error and trajectory stores states
        assert prop.trajectory.length > 0

    def test_sgppropagator_set_output_format_ecef(self, iss_tle):
        """Test setting output format to ECEF Cartesian."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_format(brahe.OrbitFrame.ECEF, brahe.OrbitRepresentation.CARTESIAN, None)
        prop.step()

        # Verify it doesn't error
        assert prop.trajectory.length > 0

    def test_sgppropagator_set_output_format_degrees(self, iss_tle):
        """Test setting output format with degrees."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_format(brahe.OrbitFrame.ECI, brahe.OrbitRepresentation.KEPLERIAN, brahe.AngleFormat.DEGREES)
        prop.step()

        # Verify it doesn't error
        assert prop.trajectory.length > 0


class TestSGPPropagatorOrbitPropagatorTrait:
    """Test SGPPropagator OrbitPropagator trait methods."""

    def test_sgppropagator_orbitpropagator_step(self, iss_tle):
        """Test step method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch

        prop.step()

        current_epoch = prop.current_epoch
        # Use approximate comparison for epoch (floating point precision)
        assert abs((current_epoch - initial_epoch) - 60.0) < 0.01
        assert prop.trajectory.length == 2  # Initial + 1 step

    def test_sgppropagator_orbitpropagator_step_by(self, iss_tle):
        """Test step_by method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch

        prop.step_by(120.0)

        current_epoch = prop.current_epoch
        # Use approximate comparison for epoch (floating point precision)
        assert abs((current_epoch - initial_epoch) - 120.0) < 0.01

    def test_sgppropagator_orbitpropagator_propagate_to(self, iss_tle):
        """Test propagate_to method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch
        target_epoch = initial_epoch + 300.0

        prop.propagate_to(target_epoch)

        # Should have propagated with step size 60.0
        assert prop.trajectory.length > 1
        # Current epoch should be at or past target
        assert prop.current_epoch >= target_epoch

    def test_sgppropagator_orbitpropagator_current_state(self, iss_tle):
        """Test current state via propagator."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        state = prop.current_state()

        assert len(state) == 6
        assert all(np.isfinite(state))

    def test_sgppropagator_orbitpropagator_current_epoch(self, iss_tle):
        """Test current_epoch property."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch

        current_epoch = prop.current_epoch

        assert current_epoch == initial_epoch

    def test_sgppropagator_orbitpropagator_initial_state(self, iss_tle):
        """Test initial state via epoch property."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        initial_state = prop.state(prop.epoch)

        assert len(initial_state) == 6
        assert all(np.isfinite(initial_state))

    def test_sgppropagator_orbitpropagator_initial_epoch(self, iss_tle):
        """Test initial epoch via epoch property."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        initial_epoch = prop.epoch

        assert initial_epoch.year() == 2008

    def test_sgppropagator_orbitpropagator_step_size(self, iss_tle):
        """Test step_size property."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        assert prop.step_size == 60.0

    def test_sgppropagator_orbitpropagator_set_step_size(self, iss_tle):
        """Test set_step_size via property setter."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.step_size = 120.0

        assert prop.step_size == 120.0

    def test_sgppropagator_orbitpropagator_reset(self, iss_tle):
        """Test reset method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch

        prop.step()
        prop.step()
        prop.reset()

        assert prop.current_epoch == initial_epoch
        assert prop.trajectory.length == 1  # Only initial state

    def test_sgppropagator_orbitpropagator_trajectory(self, iss_tle):
        """Test trajectory property."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        traj = prop.trajectory

        assert traj.length == 1  # Initial state


class TestSGPPropagatorAnalyticPropagatorTrait:
    """Test SGPPropagator AnalyticPropagator trait methods."""

    def test_sgppropagator_analyticpropagator_state(self, iss_tle):
        """Test state method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        epoch = prop.epoch + 120.0

        state = prop.state(epoch)

        assert len(state) == 6
        assert all(np.isfinite(state))

    def test_sgppropagator_analyticpropagator_state_eci(self, iss_tle):
        """Test state_eci method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        epoch = prop.epoch

        state = prop.state_eci(epoch)

        assert len(state) == 6
        assert all(np.isfinite(state))
        # Position should be in order of Earth radius (in meters)
        pos_norm = np.linalg.norm(state[:3])
        assert 6.3e6 < pos_norm < 7.0e6

    def test_sgppropagator_analyticpropagator_state_ecef(self, iss_tle):
        """Test state_ecef method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        epoch = prop.epoch

        state_ecef = prop.state_ecef(epoch)

        assert len(state_ecef) == 6
        assert all(np.isfinite(state_ecef))

        # ECEF state should be different from ECI due to frame rotation
        state_eci = prop.state_eci(epoch)
        diff_norm = np.linalg.norm(state_ecef - state_eci)
        assert diff_norm > 0.0

    def test_sgppropagator_analyticpropagator_states(self, iss_tle):
        """Test states method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch
        epochs = [initial_epoch + i * 60.0 for i in range(5)]

        traj = prop.states(epochs)

        assert len(traj) == 5
        for i in range(5):
            state = traj[i]
            print(state)
            assert len(state) == 6
            assert all(np.isfinite(state))

    def test_sgppropagator_analyticpropagator_states_eci(self, iss_tle):
        """Test states_eci method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch
        epochs = [initial_epoch + i * 60.0 for i in range(5)]

        traj = prop.states_eci(epochs)

        assert len(traj) == 5
        for i in range(5):
            state = traj[i]
            assert len(state) == 6
            assert all(np.isfinite(state))


class TestOldBraheTLEFunctions:
    """Test standalone TLE utility functions."""

    def test_calculate_tle_line_checksum(self):
        """Test TLE line checksum calculation."""
        # Test with line 1
        checksum = brahe.calculate_tle_line_checksum(ISS_LINE1)
        assert checksum == 7

        # Test with line 2
        checksum = brahe.calculate_tle_line_checksum(ISS_LINE2)
        assert checksum == 7

    def test_validate_tle_line_valid(self):
        """Test TLE line validation with valid lines."""
        assert brahe.validate_tle_line(ISS_LINE1) == True
        assert brahe.validate_tle_line(ISS_LINE2) == True

    def test_validate_tle_line_invalid(self):
        """Test TLE line validation with invalid checksum."""
        # Change last digit (checksum) to make it invalid
        invalid_line = ISS_LINE1[:-1] + '6'  # Change checksum from 7 to 6
        assert brahe.validate_tle_line(invalid_line) == False

    def test_validate_tle_lines(self):
        """Test TLE line pair validation."""
        assert brahe.validate_tle_lines(ISS_LINE1, ISS_LINE2) == True

    def test_keplerian_elements_from_tle(self, eop_original_brahe):
        """Test extracting Keplerian elements from TLE."""
        epoch, elements = brahe.keplerian_elements_from_tle(ISS_LINE1, ISS_LINE2)

        assert len(elements) == 6
        # Elements are [a, e, i, raan, argp, M]
        # Values extracted from ISS TLE
        assert elements[0] == pytest.approx(6730960.675248184, abs=1.0)  # Semi-major axis in meters
        assert elements[1] == pytest.approx(0.0006703, abs=1e-7)  # Eccentricity
        assert elements[2] == pytest.approx(51.6416, abs=1e-4)  # InJclination (degrees)
        assert elements[3] == pytest.approx(247.4627, abs=1e-4)  # RAAN (degrees)
        assert elements[4] == pytest.approx(130.536, abs=1e-4)  # Argument of periapsis (degrees)
        assert elements[5] == pytest.approx(325.0288, abs=1e-4)  # Mean anomaly (degrees)

    def test_epoch_from_tle(self, eop_original_brahe):
        """Test extracting epoch from TLE."""
        epoch = brahe.epoch_from_tle(ISS_LINE1)

        assert epoch.year() == 2008
        assert epoch.month() == 9
        assert epoch.day() == 20

    def test_sgppropagator_state_teme(self, iss_tle, eop_original_brahe):
        """Test state output in TEME frame (native SGP4 output)."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        state = prop.state(prop.epoch)

        assert len(state) == 6
        # TEME is the native SGP4 output frame
        assert state[0] == pytest.approx(4083909.8260273533, abs=1e-8)
        assert state[1] == pytest.approx(-993636.8325621719, abs=1e-8)
        assert state[2] == pytest.approx(5243614.536966579, abs=1e-8)
        assert state[3] == pytest.approx(2512.831950943635, abs=1e-8)
        assert state[4] == pytest.approx(7259.8698423432315, abs=1e-8)
        assert state[5] == pytest.approx(-583.775727402632, abs=1e-8)

    def test_sgppropagator_state_pef(self, iss_tle, eop_original_brahe):
        """Test state output in PEF frame."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        state = prop.state_pef(prop.epoch)

        assert len(state) == 6
        # TEME is the native SGP4 output frame
        assert state[0] == pytest.approx(-3953205.7105210484, abs=1.5e-1)
        assert state[1] == pytest.approx(1427514.704810681, abs=1.5e-1)
        assert state[2] == pytest.approx(5243614.536966579, abs=1.5e-1)
        assert state[3] == pytest.approx(-3175.692140186211, abs=1.5e-1)
        assert state[4] == pytest.approx(-6658.887120918979, abs=1.5e-1)
        assert state[5] == pytest.approx(-583.775727402632, abs=1.5e-1)

    @pytest.mark.xfail(reason="Error is higher than expected - Need to investigate frame transformations")
    def test_sgppropagator_state_ecef(self, iss_tle, eop_original_brahe):
        """Test state output in ECEF/ITRF frame."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        state = prop.state_ecef(prop.epoch)

        assert len(state) == 6
        # ECEF/ITRF frame
        # Note: Frame transformations have looser tolerance due to EOP variations
        assert state[0] == pytest.approx( -3953198.5496517573, abs=1.5e-1)
        assert state[1] == pytest.approx( 1427508.1713723878, abs=1.5e-1)
        assert state[2] == pytest.approx( 5243621.714247745, abs=1.5e-1)
        assert state[3] == pytest.approx( -3414.313706718372, abs=1.5e-1)
        assert state[4] == pytest.approx( -7222.549343535009, abs=1.5e-1)
        assert state[5] == pytest.approx( -583.7798954042405, abs=1.5e-1)

    @pytest.mark.xfail(reason="Error is higher than expected - Need to investigate frame transformations")
    def test_sgppropagator_state_eci(self, iss_tle, eop_original_brahe):
        """Test state output in ECI/GCRF frame."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        state = prop.state_eci(prop.epoch)

        assert len(state) == 6
        # ECI/GCRF frame (after TEME -> PEF -> ECEF -> ECI conversion)
        # Note: Frame transformations have looser tolerance due to EOP variations
        assert state[0] == pytest.approx( 4086521.040536244, abs=1.5e-1)
        assert state[1] == pytest.approx( -1001422.0787863219, abs=1.5e-1)
        assert state[2] == pytest.approx( 5240097.960898061, abs=1.5e-1)
        assert state[3] == pytest.approx( 2704.171077071122, abs=1.5e-1)
        assert state[4] == pytest.approx( 7840.6666110244705, abs=1.5e-1)
        assert state[5] == pytest.approx( -586.3906587951877, abs=1.5e-1)
