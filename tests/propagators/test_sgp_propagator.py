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

        # Verify identity fields are automatically set
        assert prop.name == "ISS (ZARYA)"
        assert prop.id == 25544

    def test_sgppropagator_set_output_format_cartesian(self, iss_tle):
        """Test setting output format to ECI Cartesian."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_format(
            brahe.OrbitFrame.ECI, brahe.OrbitRepresentation.CARTESIAN, None
        )
        prop.step()

        # Verify it doesn't error and trajectory stores states
        assert prop.trajectory.length > 0

    def test_sgppropagator_set_output_format_keplerian(self, iss_tle):
        """Test setting output format to ECI Keplerian."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_format(
            brahe.OrbitFrame.ECI,
            brahe.OrbitRepresentation.KEPLERIAN,
            brahe.AngleFormat.RADIANS,
        )
        prop.step()

        # Verify it doesn't error and trajectory stores states
        assert prop.trajectory.length > 0

    def test_sgppropagator_set_output_format_ecef(self, iss_tle):
        """Test setting output format to ECEF Cartesian."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_format(
            brahe.OrbitFrame.ECEF, brahe.OrbitRepresentation.CARTESIAN, None
        )
        prop.step()

        # Verify it doesn't error
        assert prop.trajectory.length > 0

    def test_sgppropagator_set_output_format_degrees(self, iss_tle):
        """Test setting output format with degrees."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_format(
            brahe.OrbitFrame.ECI,
            brahe.OrbitRepresentation.KEPLERIAN,
            brahe.AngleFormat.DEGREES,
        )
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

        # State should have changed after propagation
        new_state = prop.current_state()
        initial_state = prop.initial_state()
        assert not np.array_equal(new_state, initial_state)

    def test_sgppropagator_orbitpropagator_step_by(self, iss_tle):
        """Test step_by method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch

        prop.step_by(120.0)

        current_epoch = prop.current_epoch
        # Use approximate comparison for epoch (floating point precision)
        assert abs((current_epoch - initial_epoch) - 120.0) < 0.01

        # Confirm only 2 states in trajectory (initial + 1 step)
        assert prop.trajectory.length == 2

        # State should have changed after propagation
        new_state = prop.current_state()
        initial_state = prop.initial_state()
        assert not np.array_equal(new_state, initial_state)

    def test_sgppropagator_orbitpropagator_propagate_steps(self, iss_tle):
        """Test propagate_steps method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch

        prop.propagate_steps(5)

        current_epoch = prop.current_epoch
        assert abs((current_epoch - initial_epoch) - 300.0) < 0.01
        assert prop.trajectory.length == 6  # Initial + 5 steps

        # State should have changed after propagation
        new_state = prop.current_state()
        initial_state = prop.initial_state()
        assert not np.array_equal(new_state, initial_state)

    def test_sgppropagator_orbitpropagator_step_past(self, iss_tle):
        """Test step_past method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch

        target_epoch = initial_epoch + 250.0
        prop.step_past(target_epoch)

        current_epoch = prop.current_epoch
        assert current_epoch > target_epoch

        # Should have 6 steps: initial + 5 steps of 60s
        assert prop.trajectory.length == 6
        assert abs((current_epoch - initial_epoch) - 300.0) < 0.01

        # State should have changed after propagation
        new_state = prop.current_state()
        initial_state = prop.initial_state()
        assert not np.array_equal(new_state, initial_state)

    def test_sgppropagator_orbitpropagator_propagate_to(self, iss_tle):
        """Test propagate_to method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch
        target_epoch = initial_epoch + 90.0  # 90 seconds forward

        prop.propagate_to(target_epoch)

        current_epoch = prop.current_epoch
        assert current_epoch == target_epoch

        # Should have 3 steps: initial + 1 step of 60s + 1 step of 30s
        assert prop.trajectory.length == 3

        # State should have changed after propagation
        new_state = prop.current_state()
        initial_state = prop.initial_state()
        assert not np.array_equal(new_state, initial_state)

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

    def test_sgppropagator_set_step_size_method(self, iss_tle):
        """Test set_step_size explicit method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        # Test explicit method call (in addition to property setter)
        prop.set_step_size(120.0)

        assert prop.step_size == 120.0

        # Test that both property and method work interchangeably
        prop.step_size = 90.0
        assert prop.step_size == 90.0

        prop.set_step_size(150.0)
        assert prop.step_size == 150.0

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

    def test_sgppropagator_orbitpropagator_set_eviction_policy_max_size(self, iss_tle):
        """Test set_eviction_policy_max_size method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        prop.set_eviction_policy_max_size(5)

        # Propagate 10 steps
        prop.propagate_steps(10)

        # Should only keep 5 states
        assert prop.trajectory.length == 5

    def test_sgppropagator_orbitpropagator_set_eviction_policy_max_age(self, iss_tle):
        """Test set_eviction_policy_max_age method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        # Set eviction policy - only keep states within 120 seconds of current
        prop.set_eviction_policy_max_age(120.0)

        # Propagate several steps (10 * 60s = 600s total)
        prop.propagate_steps(10)

        # Should have evicted old states - should keep only last ~3 states (120s / 60s step)
        # Plus current state: 3 previous + current = 4 states max
        assert prop.trajectory.length <= 4
        assert prop.trajectory.length > 0


class TestSGPPropagatorStateProviderTrait:
    """Test SGPPropagator StateProvider trait methods."""

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

    def test_sgppropagator_state_as_osculating_elements(self, iss_tle):
        """Test state_as_osculating_elements method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        epoch = prop.epoch

        # Test with radians
        elements_rad = prop.state_as_osculating_elements(
            epoch, brahe.AngleFormat.RADIANS
        )

        # Verify we got keplerian elements (all finite)
        assert len(elements_rad) == 6
        assert all(np.isfinite(elements_rad))

        # Semi-major axis should be positive and around ISS altitude
        assert elements_rad[0] > 0.0
        assert 6.3e6 < elements_rad[0] < 7.0e6  # meters

        # Eccentricity should be non-negative and small for ISS
        assert elements_rad[1] >= 0.0
        assert elements_rad[1] < 0.1

        # Inclination should be around 51.6 degrees (in radians)
        assert elements_rad[2] == pytest.approx(np.radians(51.6), abs=0.1)

        # Test with degrees
        elements_deg = prop.state_as_osculating_elements(
            epoch, brahe.AngleFormat.DEGREES
        )

        # Verify degree conversion
        assert len(elements_deg) == 6
        assert all(np.isfinite(elements_deg))

        # First two elements (a, e) should be same in both formats
        assert elements_deg[0] == pytest.approx(elements_rad[0], rel=1e-10)
        assert elements_deg[1] == pytest.approx(elements_rad[1], rel=1e-10)

        # Angular elements should be converted from radians to degrees
        for i in range(2, 6):
            assert elements_deg[i] == pytest.approx(
                np.degrees(elements_rad[i]), rel=1e-8
            )

        # Inclination should be around 51.6 degrees
        assert elements_deg[2] == pytest.approx(51.6, abs=0.1)

    def test_sgppropagator_states_as_osculating_elements(self, iss_tle):
        """Test states_as_osculating_elements method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch
        epochs = [
            initial_epoch + i * 3600.0 for i in range(5)
        ]  # Every hour for 5 hours

        # Test with degrees
        elements_list = prop.states_as_osculating_elements(
            epochs, brahe.AngleFormat.DEGREES
        )

        # Verify we got the right number of element sets
        assert len(elements_list) == 5

        # Check each element set
        for i, elements in enumerate(elements_list):
            assert len(elements) == 6
            assert all(np.isfinite(elements))

            # Semi-major axis should be positive
            assert elements[0] > 0.0
            assert 6.3e6 < elements[0] < 7.0e6  # meters

            # Eccentricity should be non-negative
            assert elements[1] >= 0.0
            assert elements[1] < 0.1

            # Inclination should be around 51.6 degrees
            assert elements[2] == pytest.approx(51.6, abs=0.5)

            # All angles should be in valid range for degrees
            for j in range(2, 6):
                assert -360.0 <= elements[j] <= 360.0

        # Test with radians
        elements_list_rad = prop.states_as_osculating_elements(
            epochs, brahe.AngleFormat.RADIANS
        )

        # Verify conversion consistency
        assert len(elements_list_rad) == 5

        for i in range(5):
            # First two elements should match
            assert elements_list_rad[i][0] == pytest.approx(
                elements_list[i][0], rel=1e-10
            )
            assert elements_list_rad[i][1] == pytest.approx(
                elements_list[i][1], rel=1e-10
            )

            # Angular elements should be converted
            for j in range(2, 6):
                assert elements_list_rad[i][j] == pytest.approx(
                    np.radians(elements_list[i][j]), rel=1e-8
                )


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
        assert brahe.validate_tle_line(ISS_LINE1) is True
        assert brahe.validate_tle_line(ISS_LINE2) is True

    def test_validate_tle_line_invalid(self):
        """Test TLE line validation with invalid checksum."""
        # Change last digit (checksum) to make it invalid
        invalid_line = ISS_LINE1[:-1] + "6"  # Change checksum from 7 to 6
        assert brahe.validate_tle_line(invalid_line) is False

    def test_validate_tle_lines(self):
        """Test TLE line pair validation."""
        assert brahe.validate_tle_lines(ISS_LINE1, ISS_LINE2) is True

    def test_keplerian_elements_from_tle(self, eop_original_brahe):
        """Test extracting Keplerian elements from TLE."""
        epoch, elements = brahe.keplerian_elements_from_tle(ISS_LINE1, ISS_LINE2)

        assert len(elements) == 6
        # Elements are [a, e, i, raan, argp, M]
        # Values extracted from ISS TLE
        assert elements[0] == pytest.approx(
            6730960.675248184, abs=1.0
        )  # Semi-major axis in meters
        assert elements[1] == pytest.approx(0.0006703, abs=1e-7)  # Eccentricity
        assert elements[2] == pytest.approx(51.6416, abs=1e-4)  # InJclination (degrees)
        assert elements[3] == pytest.approx(247.4627, abs=1e-4)  # RAAN (degrees)
        assert elements[4] == pytest.approx(
            130.536, abs=1e-4
        )  # Argument of periapsis (degrees)
        assert elements[5] == pytest.approx(
            325.0288, abs=1e-4
        )  # Mean anomaly (degrees)

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

    @pytest.mark.xfail(
        reason="Error is higher than expected - Need to investigate frame transformations"
    )
    def test_sgppropagator_state_ecef(self, iss_tle, eop_original_brahe):
        """Test state output in ECEF/ITRF frame."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        state = prop.state_ecef(prop.epoch)

        assert len(state) == 6
        # ECEF/ITRF frame
        # Note: Frame transformations have looser tolerance due to EOP variations
        assert state[0] == pytest.approx(-3953198.5496517573, abs=1.5e-1)
        assert state[1] == pytest.approx(1427508.1713723878, abs=1.5e-1)
        assert state[2] == pytest.approx(5243621.714247745, abs=1.5e-1)
        assert state[3] == pytest.approx(-3414.313706718372, abs=1.5e-1)
        assert state[4] == pytest.approx(-7222.549343535009, abs=1.5e-1)
        assert state[5] == pytest.approx(-583.7798954042405, abs=1.5e-1)

    @pytest.mark.xfail(
        reason="Error is higher than expected - Need to investigate frame transformations"
    )
    def test_sgppropagator_state_eci(self, iss_tle, eop_original_brahe):
        """Test state output in ECI/GCRF frame."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        state = prop.state_eci(prop.epoch)

        assert len(state) == 6
        # ECI/GCRF frame (after TEME -> PEF -> ECEF -> ECI conversion)
        # Note: Frame transformations have looser tolerance due to EOP variations
        assert state[0] == pytest.approx(4086521.040536244, abs=1.5e-1)
        assert state[1] == pytest.approx(-1001422.0787863219, abs=1.5e-1)
        assert state[2] == pytest.approx(5240097.960898061, abs=1.5e-1)
        assert state[3] == pytest.approx(2704.171077071122, abs=1.5e-1)
        assert state[4] == pytest.approx(7840.6666110244705, abs=1.5e-1)
        assert state[5] == pytest.approx(-586.3906587951877, abs=1.5e-1)

    def test_sgppropagator_get_elements_radians(self, iss_tle):
        """Test get_elements method with radians."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        elements = prop.get_elements(brahe.AngleFormat.RADIANS)

        # Expected values from ISS TLE
        assert elements[0] == pytest.approx(6730960.676936833, abs=1.0)  # a [m]
        assert elements[1] == pytest.approx(0.0006703, abs=1e-10)  # e
        assert elements[2] == pytest.approx(0.9013159509979036, abs=1e-10)  # i [rad]
        assert elements[3] == pytest.approx(4.319038890874972, abs=1e-10)  # raan [rad]
        assert elements[4] == pytest.approx(2.278282992383318, abs=1e-10)  # argp [rad]
        assert elements[5] == pytest.approx(5.672822723806145, abs=1e-10)  # M [rad]

    def test_sgppropagator_get_elements_degrees(self, iss_tle):
        """Test get_elements method with degrees."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        elements = prop.get_elements(brahe.AngleFormat.DEGREES)

        # Expected values from ISS TLE
        assert elements[0] == pytest.approx(6730960.676936833, abs=1.0)  # a [m]
        assert elements[1] == pytest.approx(0.0006703, abs=1e-10)  # e
        assert elements[2] == pytest.approx(51.6416, abs=1e-10)  # i [deg]
        assert elements[3] == pytest.approx(247.4627, abs=1e-10)  # raan [deg]
        assert elements[4] == pytest.approx(130.5360, abs=1e-10)  # argp [deg]
        assert elements[5] == pytest.approx(325.0288, abs=1e-10)  # M [deg]
