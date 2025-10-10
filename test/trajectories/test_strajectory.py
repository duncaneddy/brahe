"""
Tests for the STrajectory6 class in brahe.

These tests mirror the Rust STrajectory test suite to ensure Python bindings work correctly.
Note: In Python, STrajectory6 is an alias for DTrajectory with dimension=6, so these tests
verify 6-dimensional trajectory behavior specifically.
"""

import pytest
import numpy as np
import brahe


def create_test_trajectory():
    """Helper to create a test STrajectory6 with sample data."""
    epochs = [
        brahe.Epoch.from_jd(2451545.0, "UTC"),
        brahe.Epoch.from_jd(2451545.1, "UTC"),
        brahe.Epoch.from_jd(2451545.2, "UTC"),
    ]
    states = [
        np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
        np.array([7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0]),
        np.array([7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0]),
    ]
    return brahe.STrajectory6(), epochs, states


class TestSTrajectoryTrajectory:
    """Tests for Trajectory trait implementation on STrajectory6."""

    def test_strajectory_trajectory_new(self):
        """Test STrajectory6 creation."""
        trajectory = brahe.STrajectory6()

        assert len(trajectory) == 0
        assert trajectory.interpolation_method == brahe.InterpolationMethod.linear
        assert len(trajectory) == 0  # is_empty equivalent

    def test_strajectory_trajectory_add_state(self):
        """Test adding states to trajectory."""
        traj, epochs, states = create_test_trajectory()

        # Add states in order
        traj.add_state(epochs[0], states[0])
        traj.add_state(epochs[2], states[2])

        # Add a state in between
        traj.add_state(epochs[1], states[1])

        assert len(traj) == 3
        assert traj.epoch(0).jd() == 2451545.0
        assert traj.epoch(1).jd() == 2451545.1
        assert traj.epoch(2).jd() == 2451545.2

    def test_strajectory_trajectory_state_at_index(self):
        """Test retrieving state by index."""
        traj, epochs, states = create_test_trajectory()
        for epoch, state in zip(epochs, states):
            traj.add_state(epoch, state)

        # Test valid indices
        state0 = traj.state(0)
        assert state0[0] == 7000e3

        state1 = traj.state(1)
        assert state1[0] == 7100e3

        state2 = traj.state(2)
        assert state2[0] == 7200e3

        # Test invalid index
        with pytest.raises(Exception):
            traj.state(10)

    def test_strajectory_trajectory_epoch_at_index(self):
        """Test retrieving epoch by index."""
        traj, epochs, states = create_test_trajectory()
        for epoch, state in zip(epochs, states):
            traj.add_state(epoch, state)

        # Test valid indices
        epoch0 = traj.epoch(0)
        assert epoch0.jd() == 2451545.0

        epoch1 = traj.epoch(1)
        assert epoch1.jd() == 2451545.1

        epoch2 = traj.epoch(2)
        assert epoch2.jd() == 2451545.2

        # Test invalid index
        with pytest.raises(Exception):
            traj.epoch(10)

    def test_strajectory_trajectory_nearest_state(self):
        """Test finding nearest state to query epoch."""
        traj, epochs, states = create_test_trajectory()
        for epoch, state in zip(epochs, states):
            traj.add_state(epoch, state)

        # Test finding nearest to exact epoch
        query_epoch = epochs[1]
        nearest_epoch, nearest_state = traj.nearest_state(query_epoch)
        assert nearest_epoch.jd() == epochs[1].jd()
        assert nearest_state[0] == 7100e3

        # Test finding nearest to mid-point (closer to second epoch)
        mid_epoch = brahe.Epoch.from_jd(2451545.06, "UTC")
        nearest_epoch, _ = traj.nearest_state(mid_epoch)
        assert nearest_epoch.jd() == epochs[1].jd()

    def test_strajectory_trajectory_len(self):
        """Test trajectory length."""
        traj = brahe.STrajectory6()

        assert len(traj) == 0

        epoch = brahe.Epoch.from_jd(2451545.0, "UTC")
        state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add_state(epoch, state)

        assert len(traj) == 1

    def test_strajectory_trajectory_is_empty(self):
        """Test empty trajectory check."""
        traj = brahe.STrajectory6()

        assert len(traj) == 0

        epoch = brahe.Epoch.from_jd(2451545.0, "UTC")
        state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add_state(epoch, state)

        assert len(traj) > 0

    def test_strajectory_trajectory_start_epoch(self):
        """Test getting start epoch."""
        traj = brahe.STrajectory6()

        assert traj.start_epoch is None

        epoch = brahe.Epoch.from_jd(2451545.0, "UTC")
        state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add_state(epoch, state)

        assert traj.start_epoch.jd() == epoch.jd()

    def test_strajectory_trajectory_end_epoch(self):
        """Test getting end epoch."""
        traj = brahe.STrajectory6()

        assert traj.end_epoch is None

        epoch1 = brahe.Epoch.from_jd(2451545.0, "UTC")
        epoch2 = brahe.Epoch.from_jd(2451545.1, "UTC")
        state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add_state(epoch1, state)
        traj.add_state(epoch2, state)

        assert traj.end_epoch.jd() == epoch2.jd()

    def test_strajectory_trajectory_timespan(self):
        """Test trajectory timespan calculation."""
        traj, epochs, states = create_test_trajectory()
        for i in range(2):  # Just add first two
            traj.add_state(epochs[i], states[i])

        timespan = traj.time_span
        assert abs(timespan - 0.1 * 86400.0) < 1e-5

    def test_strajectory_trajectory_first(self):
        """Test getting first state."""
        traj, epochs, states = create_test_trajectory()
        for i in range(2):
            traj.add_state(epochs[i], states[i])

        first_epoch, first_state = traj.first()
        assert first_epoch.jd() == epochs[0].jd()
        assert np.array_equal(first_state, states[0])

    def test_strajectory_trajectory_last(self):
        """Test getting last state."""
        traj, epochs, states = create_test_trajectory()
        for i in range(2):
            traj.add_state(epochs[i], states[i])

        last_epoch, last_state = traj.last()
        assert last_epoch.jd() == epochs[1].jd()
        assert np.array_equal(last_state, states[1])

    def test_strajectory_trajectory_clear(self):
        """Test clearing trajectory."""
        traj = brahe.STrajectory6()

        epoch = brahe.Epoch.from_jd(2451545.0, "UTC")
        state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add_state(epoch, state)

        assert len(traj) == 1
        traj.clear()
        assert len(traj) == 0

    def test_strajectory_trajectory_remove_state(self):
        """Test removing state by epoch."""
        traj, epochs, states = create_test_trajectory()
        for i in range(2):
            traj.add_state(epochs[i], states[i])

        removed_state = traj.remove_state(epochs[0])
        assert removed_state[0] == 7000e3
        assert len(traj) == 1

    def test_strajectory_trajectory_remove_state_at_index(self):
        """Test removing state by index."""
        traj, epochs, states = create_test_trajectory()
        for i in range(2):
            traj.add_state(epochs[i], states[i])

        removed_epoch, removed_state = traj.remove_state_at_index(0)
        assert removed_epoch.jd() == 2451545.0
        assert removed_state[0] == 7000e3
        assert len(traj) == 1

    def test_strajectory_trajectory_get(self):
        """Test getting state and epoch by index."""
        traj, epochs, states = create_test_trajectory()
        for i in range(2):
            traj.add_state(epochs[i], states[i])

        epoch, state = traj.get(1)
        assert epoch.jd() == 2451545.1
        assert state[0] == 7100e3

    def test_strajectory_state_at_epoch_errors(self):
        """Test state_at_epoch error conditions."""
        traj, epochs, states = create_test_trajectory()
        for epoch, state in zip(epochs, states):
            traj.add_state(epoch, state)

        too_early = brahe.Epoch.from_jd(2451544.0, "UTC")
        with pytest.raises(Exception):
            traj.state_at_epoch(too_early)

        too_late = brahe.Epoch.from_jd(2451546.0, "UTC")
        with pytest.raises(Exception):
            traj.state_at_epoch(too_late)

    def test_strajectory_timespan_edge_cases(self):
        """Test timespan edge cases."""
        traj = brahe.STrajectory6()

        # Empty trajectory
        assert traj.time_span is None

        # Single state
        epoch = brahe.Epoch.from_jd(2451545.0, "UTC")
        state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add_state(epoch, state)
        assert traj.time_span is None or traj.time_span == 0.0

    def test_strajectory_trajectory_index_before_epoch(self):
        """Test finding index before epoch."""
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        t1 = t0 + 60.0  # t0 + 60 seconds
        t2 = t0 + 120.0  # t0 + 120 seconds

        traj = brahe.STrajectory6()
        states = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            np.array([21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
        ]

        traj.add_state(t0, states[0])
        traj.add_state(t1, states[1])
        traj.add_state(t2, states[2])

        # Test finding index before t0 (should error - before all states)
        before_t0 = t0 + (-10.0)
        with pytest.raises(Exception):
            traj.index_before_epoch(before_t0)

        # Test finding index before t0+30s (should return index 0)
        t0_plus_30 = t0 + 30.0
        assert traj.index_before_epoch(t0_plus_30) == 0

        # Test finding index before t1 (should return index 1 - exact match)
        assert traj.index_before_epoch(t1) == 1

    def test_strajectory_trajectory_index_after_epoch(self):
        """Test finding index after epoch."""
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        t1 = t0 + 60.0  # t0 + 60 seconds
        t2 = t0 + 120.0  # t0 + 120 seconds

        traj = brahe.STrajectory6()
        states = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            np.array([21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
        ]

        traj.add_state(t0, states[0])
        traj.add_state(t1, states[1])
        traj.add_state(t2, states[2])

        # Test finding index after t0 (should return index 0 - exact match)
        assert traj.index_after_epoch(t0) == 0

        # Test finding index after t0+30s (should return index 1)
        t0_plus_30 = t0 + 30.0
        assert traj.index_after_epoch(t0_plus_30) == 1

        # Test finding index after t2+150s (should error - after all states)
        t2_plus_150 = t0 + 270.0
        with pytest.raises(Exception):
            traj.index_after_epoch(t2_plus_150)

    def test_strajectory_trajectory_state_before_epoch(self):
        """Test getting state before epoch."""
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        t1 = t0 + 60.0  # t0 + 60 seconds
        t2 = t0 + 120.0  # t0 + 120 seconds

        traj = brahe.STrajectory6()
        states = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            np.array([21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
        ]

        traj.add_state(t0, states[0])
        traj.add_state(t1, states[1])
        traj.add_state(t2, states[2])

        # Test that state_before_epoch returns correct (epoch, state) tuples
        t0_plus_30 = t0 + 30.0
        epoch, state = traj.state_before_epoch(t0_plus_30)
        assert epoch.jd() == t0.jd()
        assert abs(state[0] - 1.0) < 1e-10

    def test_strajectory_trajectory_state_after_epoch(self):
        """Test getting state after epoch."""
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        t1 = t0 + 60.0  # t0 + 60 seconds
        t2 = t0 + 120.0  # t0 + 120 seconds

        traj = brahe.STrajectory6()
        states = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            np.array([21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
        ]

        traj.add_state(t0, states[0])
        traj.add_state(t1, states[1])
        traj.add_state(t2, states[2])

        # Test that state_after_epoch returns correct (epoch, state) tuples
        t0_plus_30 = t0 + 30.0
        epoch, state = traj.state_after_epoch(t0_plus_30)
        assert epoch.jd() == t1.jd()
        assert abs(state[0] - 11.0) < 1e-10


class TestSTrajectoryInterpolatable:
    """Tests for Interpolatable trait implementation on STrajectory6."""

    def test_strajectory_interpolatable_get_interpolation_method(self):
        """Test getting interpolation method."""
        traj = brahe.STrajectory6()

        # Test that get_interpolation_method returns Linear
        assert traj.interpolation_method == brahe.InterpolationMethod.linear

        # Set it to different methods and verify
        traj.set_interpolation_method(brahe.InterpolationMethod.linear)
        assert traj.interpolation_method == brahe.InterpolationMethod.linear

        traj.set_interpolation_method(brahe.InterpolationMethod.linear)
        assert traj.interpolation_method == brahe.InterpolationMethod.linear

    def test_strajectory_interpolatable_interpolate_linear(self):
        """Test linear interpolation."""
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        t1 = t0 + 60.0  # t0 + 60 seconds
        t2 = t0 + 120.0  # t0 + 120 seconds

        traj = brahe.STrajectory6()
        states = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([60.0, 120.0, 180.0, 240.0, 300.0, 360.0]),
            np.array([120.0, 240.0, 360.0, 480.0, 600.0, 720.0]),
        ]

        traj.add_state(t0, states[0])
        traj.add_state(t1, states[1])
        traj.add_state(t2, states[2])

    def test_strajectory_interpolatable_interpolate(self):
        """Test generic interpolate method."""
        t0 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        t1 = t0 + 60.0  # t0 + 60 seconds
        t2 = t0 + 120.0  # t0 + 120 seconds

        traj = brahe.STrajectory6()
        states = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([60.0, 120.0, 180.0, 240.0, 300.0, 360.0]),
            np.array([120.0, 240.0, 360.0, 480.0, 600.0, 720.0]),
        ]

        traj.add_state(t0, states[0])
        traj.add_state(t1, states[1])
        traj.add_state(t2, states[2])


class TestSTrajectoryIndex:
    """Tests for Index trait implementation (Python __getitem__)."""

    def test_strajectory_index(self):
        """Test indexing into trajectory."""
        traj, epochs, states = create_test_trajectory()

        for epoch, state in zip(epochs, states):
            traj.add_state(epoch, state)

        # Test positive indexing
        state0 = traj[0]
        assert abs(state0[0] - 7000e3) < 1.0

        state1 = traj[1]
        assert abs(state1[0] - 7100e3) < 1.0

        state2 = traj[2]
        assert abs(state2[0] - 7200e3) < 1.0

    def test_strajectory_index_negative(self):
        """Test negative indexing into trajectory."""
        traj, epochs, states = create_test_trajectory()

        for epoch, state in zip(epochs, states):
            traj.add_state(epoch, state)

        # Test negative indexing
        state_last = traj[-1]
        assert abs(state_last[0] - 7200e3) < 1.0

        state_second_last = traj[-2]
        assert abs(state_second_last[0] - 7100e3) < 1.0

    def test_strajectory_index_out_of_bounds(self):
        """Test indexing out of bounds raises IndexError."""
        traj, epochs, states = create_test_trajectory()

        for epoch, state in zip(epochs, states):
            traj.add_state(epoch, state)

        with pytest.raises(IndexError):
            _ = traj[10]

        with pytest.raises(IndexError):
            _ = traj[-10]


class TestSTrajectoryIterator:
    """Tests for Iterator trait implementation (Python __iter__)."""

    def test_strajectory_iterator(self):
        """Test iterating over trajectory yields (epoch, state) pairs."""
        traj, epochs, states = create_test_trajectory()

        for epoch, state in zip(epochs, states):
            traj.add_state(epoch, state)

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

    def test_strajectory_iterator_empty(self):
        """Test iterating over empty trajectory."""
        traj = brahe.STrajectory6()

        count = 0
        for _ in traj:
            count += 1

        assert count == 0
