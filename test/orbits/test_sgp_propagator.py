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

    def test_sgppropagator_set_output_cartesian(self, iss_tle):
        """Test setting output representation to Cartesian."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_cartesian()
        prop.step()

        # Verify it doesn't error and trajectory stores states
        assert prop.trajectory.length > 0

    def test_sgppropagator_set_output_keplerian(self, iss_tle):
        """Test setting output representation to Keplerian."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_keplerian()
        prop.step()

        # Verify it doesn't error and trajectory stores states
        assert prop.trajectory.length > 0

    def test_sgppropagator_set_output_frame(self, iss_tle):
        """Test setting output frame to ECEF."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_frame(brahe.OrbitFrame.ecef)
        prop.step()

        # Verify it doesn't error
        assert prop.trajectory.length > 0

    def test_sgppropagator_set_output_angle_format(self, iss_tle):
        """Test setting output angle format."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        prop.set_output_angle_format(brahe.AngleFormat.degrees)
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
        """Test current state via trajectory."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)

        state = prop.trajectory.current_state_vector()

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

        assert traj.length == 5
        for i in range(5):
            state = traj.state_at_index(i)
            assert len(state) == 6
            assert all(np.isfinite(state))

    def test_sgppropagator_analyticpropagator_states_eci(self, iss_tle):
        """Test states_eci method."""
        prop = brahe.SGPPropagator.from_tle(iss_tle[0], iss_tle[1], 60.0)
        initial_epoch = prop.epoch
        epochs = [initial_epoch + i * 60.0 for i in range(5)]

        traj = prop.states_eci(epochs)

        assert traj.length == 5
        for i in range(5):
            state = traj.state_at_index(i)
            assert len(state) == 6
            assert all(np.isfinite(state))
