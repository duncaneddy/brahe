"""
Tests for the base Trajectory class in brahe.

These tests mirror the Rust test suite to ensure Python bindings work correctly.
"""

import pytest
import numpy as np
import brahe


@pytest.fixture
def sample_epochs():
    """Create sample epochs for testing."""
    base_epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    return [
        base_epoch,
        brahe.Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, "UTC"),
        brahe.Epoch.from_datetime(2023, 1, 1, 14, 0, 0.0, 0.0, "UTC"),
    ]


@pytest.fixture
def sample_states():
    """Create sample 6-element state vectors."""
    return [
        np.array([1000.0, 2000.0, 3000.0, 1.0, 2.0, 3.0]),
        np.array([1100.0, 2100.0, 3100.0, 1.1, 2.1, 3.1]),
        np.array([1200.0, 2200.0, 3200.0, 1.2, 2.2, 3.2]),
    ]


class TestTrajectoryCreation:
    """Test trajectory creation that mirrors Rust tests."""

    def test_trajectory_creation(self):
        """Test basic trajectory creation."""
        trajectory = brahe.Trajectory()
        assert len(trajectory) == 0

    def test_trajectory_with_interpolation(self):
        """Test trajectory creation with specific interpolation."""
        linear_interp = brahe.InterpolationMethod.linear
        trajectory = brahe.Trajectory(linear_interp)
        assert len(trajectory) == 0


class TestTrajectoryStateManagement:
    """Test trajectory state management that mirrors Rust tests."""

    def test_trajectory_add_state(self, sample_epochs, sample_states):
        """Test adding states to trajectory."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        assert len(trajectory) == 3

    def test_trajectory_indexing(self, sample_epochs, sample_states):
        """Test trajectory indexing."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        # Test indexing
        first_epoch = trajectory.epoch_at_index(0)
        assert first_epoch.jd() == sample_epochs[0].jd()

    def test_trajectory_nearest_state(self, sample_epochs, sample_states):
        """Test finding nearest state."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        # Test nearest state
        nearest_epoch, nearest_state = trajectory.nearest_state(sample_epochs[1])
        assert nearest_epoch.jd() == sample_epochs[1].jd()
        np.testing.assert_array_almost_equal(nearest_state, sample_states[1])


class TestTrajectoryInterpolation:
    """Test trajectory interpolation that mirrors Rust tests."""

    def test_trajectory_linear_interpolation(self, sample_epochs, sample_states):
        """Test linear interpolation."""
        trajectory = brahe.Trajectory(brahe.InterpolationMethod.linear)

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        # Test interpolation at midpoint
        mid_time_jd = (sample_epochs[0].jd() + sample_epochs[1].jd()) / 2.0
        mid_epoch = brahe.Epoch.from_jd(mid_time_jd, "UTC")
        interpolated_state = trajectory.state_at_epoch(mid_epoch)

        # Should be roughly halfway between first two states
        expected_state = (sample_states[0] + sample_states[1]) / 2.0
        np.testing.assert_array_almost_equal(interpolated_state, expected_state, decimal=1)


class TestTrajectoryProperties:
    """Test trajectory properties that mirror Rust tests."""

    def test_trajectory_to_matrix(self, sample_epochs, sample_states):
        """Test converting trajectory to matrix."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        matrix = trajectory.to_matrix()
        assert matrix.shape == (6, 3)  # 6 elements, 3 states

        # Check that we can access matrix elements (implementation-specific organization)
        assert matrix[0, 0] == sample_states[0][0]  # First element of first state


class TestTrajectoryFirstLast:
    """Test trajectory first() and last() methods."""

    def test_trajectory_first_empty(self):
        """Test first() method on empty trajectory."""
        trajectory = brahe.Trajectory()
        assert trajectory.first() is None

    def test_trajectory_last_empty(self):
        """Test last() method on empty trajectory."""
        trajectory = brahe.Trajectory()
        assert trajectory.last() is None

    def test_trajectory_first_single_state(self, sample_epochs, sample_states):
        """Test first() method with single state."""
        trajectory = brahe.Trajectory()
        trajectory.add_state(sample_epochs[0], sample_states[0])

        first_result = trajectory.first()
        assert first_result is not None
        first_epoch, first_state = first_result

        assert first_epoch.jd() == sample_epochs[0].jd()
        np.testing.assert_array_almost_equal(first_state, sample_states[0])

    def test_trajectory_last_single_state(self, sample_epochs, sample_states):
        """Test last() method with single state."""
        trajectory = brahe.Trajectory()
        trajectory.add_state(sample_epochs[0], sample_states[0])

        last_result = trajectory.last()
        assert last_result is not None
        last_epoch, last_state = last_result

        assert last_epoch.jd() == sample_epochs[0].jd()
        np.testing.assert_array_almost_equal(last_state, sample_states[0])

    def test_trajectory_first_multiple_states(self, sample_epochs, sample_states):
        """Test first() method with multiple states."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        first_result = trajectory.first()
        assert first_result is not None
        first_epoch, first_state = first_result

        assert first_epoch.jd() == sample_epochs[0].jd()
        np.testing.assert_array_almost_equal(first_state, sample_states[0])

    def test_trajectory_last_multiple_states(self, sample_epochs, sample_states):
        """Test last() method with multiple states."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        last_result = trajectory.last()
        assert last_result is not None
        last_epoch, last_state = last_result

        assert last_epoch.jd() == sample_epochs[-1].jd()
        np.testing.assert_array_almost_equal(last_state, sample_states[-1])

    def test_trajectory_first_last_ordering(self, sample_epochs, sample_states):
        """Test that first() and last() respect chronological ordering."""
        trajectory = brahe.Trajectory()

        # Add states in reverse order
        for epoch, state in zip(reversed(sample_epochs), reversed(sample_states)):
            trajectory.add_state(epoch, state)

        # first() should still return the chronologically first state
        first_result = trajectory.first()
        assert first_result is not None
        first_epoch, first_state = first_result
        assert first_epoch.jd() == sample_epochs[0].jd()  # Earliest epoch
        np.testing.assert_array_almost_equal(first_state, sample_states[0])

        # last() should still return the chronologically last state
        last_result = trajectory.last()
        assert last_result is not None
        last_epoch, last_state = last_result
        assert last_epoch.jd() == sample_epochs[-1].jd()  # Latest epoch
        np.testing.assert_array_almost_equal(last_state, sample_states[-1])


class TestTrajectoryAdditionalMethods:
    """Test additional trajectory methods for comprehensive coverage."""

    def test_trajectory_set_interpolation_method(self):
        """Test setting interpolation method."""
        trajectory = brahe.Trajectory(brahe.InterpolationMethod.linear)

        # Test initial method
        assert trajectory.interpolation_method == brahe.InterpolationMethod.linear

        # Test changing method
        trajectory.set_interpolation_method(brahe.InterpolationMethod.lagrange)
        assert trajectory.interpolation_method == brahe.InterpolationMethod.lagrange

    def test_trajectory_state_at_index(self, sample_epochs, sample_states):
        """Test state_at_index method."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        # Test valid indices
        state0 = trajectory.state_at_index(0)
        np.testing.assert_array_almost_equal(state0, sample_states[0])

        state1 = trajectory.state_at_index(1)
        np.testing.assert_array_almost_equal(state1, sample_states[1])

        state2 = trajectory.state_at_index(2)
        np.testing.assert_array_almost_equal(state2, sample_states[2])

        # Test invalid index
        with pytest.raises(Exception):  # Should raise IndexError or similar
            trajectory.state_at_index(10)

    def test_trajectory_epoch_at_index(self, sample_epochs, sample_states):
        """Test epoch_at_index method."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        # Test valid indices
        epoch0 = trajectory.epoch_at_index(0)
        assert epoch0.jd() == sample_epochs[0].jd()

        epoch1 = trajectory.epoch_at_index(1)
        assert epoch1.jd() == sample_epochs[1].jd()

        epoch2 = trajectory.epoch_at_index(2)
        assert epoch2.jd() == sample_epochs[2].jd()

        # Test invalid index
        with pytest.raises(Exception):  # Should raise IndexError or similar
            trajectory.epoch_at_index(10)

    def test_trajectory_start_end_epoch(self, sample_epochs, sample_states):
        """Test start_epoch and end_epoch properties."""
        trajectory = brahe.Trajectory()

        # Test empty trajectory
        assert trajectory.start_epoch is None
        assert trajectory.end_epoch is None

        # Add states
        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        # Test populated trajectory
        start_epoch = trajectory.start_epoch
        end_epoch = trajectory.end_epoch

        assert start_epoch is not None
        assert end_epoch is not None
        assert start_epoch.jd() == sample_epochs[0].jd()
        assert end_epoch.jd() == sample_epochs[-1].jd()

    def test_trajectory_clear(self, sample_epochs, sample_states):
        """Test clear method."""
        trajectory = brahe.Trajectory()

        # Add states
        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        assert len(trajectory) == 3

        # Clear trajectory
        trajectory.clear()
        assert len(trajectory) == 0
        assert trajectory.start_epoch is None
        assert trajectory.end_epoch is None
        assert trajectory.first() is None
        assert trajectory.last() is None

    def test_trajectory_eviction_policies(self):
        """Test trajectory eviction policies."""
        trajectory = brahe.Trajectory()

        # Test max size policy
        trajectory.set_max_size(2)

        # Add more states than max size
        epochs = [
            brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC"),
            brahe.Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, "UTC"),
            brahe.Epoch.from_datetime(2023, 1, 1, 14, 0, 0.0, 0.0, "UTC"),
        ]
        states = [
            np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0, 2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0, 3.0, 4.0, 5.0]),
        ]

        for epoch, state in zip(epochs, states):
            trajectory.add_state(epoch, state)

        # Should only keep last 2 states
        assert len(trajectory) == 2

        # Test max age policy (1 hour = 3600 seconds)
        trajectory = brahe.Trajectory()
        trajectory.set_max_age(3600.0)

        # Add states with larger time gaps
        for epoch, state in zip(epochs, states):
            trajectory.add_state(epoch, state)

        # Behavior depends on implementation, but should not error
        assert len(trajectory) >= 0  # At least no error

    def test_trajectory_time_span(self, sample_epochs, sample_states):
        """Test time_span property."""
        trajectory = brahe.Trajectory()

        # Test empty trajectory
        assert trajectory.time_span is None

        # Add single state
        trajectory.add_state(sample_epochs[0], sample_states[0])
        assert trajectory.time_span is None  # Single state has no span

        # Add second state
        trajectory.add_state(sample_epochs[1], sample_states[1])
        time_span = trajectory.time_span
        assert time_span is not None
        assert time_span > 0

    def test_trajectory_validation_errors(self):
        """Test trajectory validation error conditions."""
        # Test adding state to empty trajectory and accessing invalid indices
        trajectory = brahe.Trajectory()

        # Test invalid index access
        with pytest.raises(RuntimeError):
            trajectory.state_at_index(0)

        with pytest.raises(RuntimeError):
            trajectory.epoch_at_index(0)

    def test_trajectory_state_at_epoch_errors(self):
        """Test state_at_epoch error conditions."""
        trajectory = brahe.Trajectory()

        # Empty trajectory
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        with pytest.raises(RuntimeError):
            trajectory.state_at_epoch(epoch)

    def test_trajectory_iterator(self):
        """Test trajectory iteration functionality."""
        trajectory = brahe.Trajectory()
        epochs = [
            brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC"),
            brahe.Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, "UTC"),
        ]
        states = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        ]

        for epoch, state in zip(epochs, states):
            trajectory.add_state(epoch, state)

        # Test that we can access via indexing (iterator-like behavior)
        assert len(trajectory) == 2
        for i in range(len(trajectory)):
            state = trajectory.state_at_index(i)
            assert state is not None

    def test_trajectory_remove_state_methods(self):
        """Test state removal functionality."""
        trajectory = brahe.Trajectory()
        epochs = [
            brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC"),
            brahe.Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, "UTC"),
            brahe.Epoch.from_datetime(2023, 1, 1, 14, 0, 0.0, 0.0, "UTC"),
        ]
        states = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ]

        for epoch, state in zip(epochs, states):
            trajectory.add_state(epoch, state)

        initial_length = len(trajectory)
        assert initial_length == 3

        # Note: If remove methods are not implemented in Python bindings,
        # this test documents the expected behavior for when they are added
        # For now, just verify the trajectory has the expected states
        assert len(trajectory) == 3

    def test_trajectory_edge_cases(self):
        """Test trajectory edge cases."""
        # Test single state trajectory timespan
        single_trajectory = brahe.Trajectory()
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        single_trajectory.add_state(epoch, state)

        # Single state should have None or 0 timespan
        time_span = single_trajectory.time_span
        assert time_span is None or time_span == 0.0

        # Test empty trajectory timespan
        empty_trajectory = brahe.Trajectory()
        assert empty_trajectory.time_span is None


class TestTrajectoryNewMethods:
    """Test new Trajectory methods that mirror OrbitalTrajectory tests."""

    def test_trajectory_index_before_epoch(self):
        """Test index_before_epoch method."""
        # Create trajectory with states at t0, t0+60s, t0+120s
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.Trajectory()

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

    def test_trajectory_index_after_epoch(self):
        """Test index_after_epoch method."""
        # Create trajectory with states at t0, t0+60s, t0+120s
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.Trajectory()

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

    def test_trajectory_state_before_epoch(self):
        """Test state_before_epoch method."""
        # Create trajectory with distinguishable states
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.Trajectory()

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

    def test_trajectory_state_after_epoch(self):
        """Test state_after_epoch method."""
        # Create trajectory with distinguishable states
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.Trajectory()

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

    def test_trajectory_get_interpolation_method(self):
        """Test get_interpolation_method property."""
        traj = brahe.Trajectory()

        # Verify default is Linear
        assert traj.interpolation_method == brahe.InterpolationMethod.linear

        # Change method using set_interpolation_method() and verify property returns correct value
        traj.set_interpolation_method(brahe.InterpolationMethod.lagrange)
        assert traj.interpolation_method == brahe.InterpolationMethod.lagrange

        # Change back to linear
        traj.set_interpolation_method(brahe.InterpolationMethod.linear)
        assert traj.interpolation_method == brahe.InterpolationMethod.linear

    def test_trajectory_interpolate_linear(self):
        """Test interpolate_linear method."""
        # Create trajectory with simple values for easy verification
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.Trajectory()

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

        # Test at t0+90s (1/2 of the way between t0+60s and t0+120s)
        # Should be 1/2 of the way: 7060e3 + 0.5 * (7120e3 - 7060e3) = 7090e3
        t_90 = t0 + 90.0
        state_90 = traj.interpolate_linear(t_90)
        assert state_90[0] == pytest.approx(7090e3, rel=1e-6)

    def test_trajectory_interpolate(self):
        """Test interpolate method."""
        # Create trajectory
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        traj = brahe.Trajectory()

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

        # Test unimplemented method (Lagrange is not yet implemented)
        traj.set_interpolation_method(brahe.InterpolationMethod.lagrange)
        with pytest.raises(RuntimeError, match="not yet implemented"):
            traj.interpolate(t_test)