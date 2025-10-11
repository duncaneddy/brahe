"""
Tests for STrajectory6 in brahe.

These tests mirror the Rust STrajectory test suite to ensure 1:1 parity.
Each test corresponds to a specific Rust test in src/trajectories/strajectory.rs.
"""

import pytest
import numpy as np
from brahe import Epoch, STrajectory6, InterpolationMethod


def create_test_trajectory():
    """Helper to create a test STrajectory6 with sample data.

    Corresponds to Rust function: create_test_trajectory()
    """
    epochs = [
        Epoch.from_jd(2451545.0, "UTC"),
        Epoch.from_jd(2451545.1, "UTC"),
        Epoch.from_jd(2451545.2, "UTC"),
    ]

    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
        7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0,
    ])

    return STrajectory6.from_data(epochs, states)


# STrajectory Trait Tests

def test_strajectory_strajectory_new():
    """Rust test: test_strajectory_strajectory_new"""
    trajectory = STrajectory6()

    assert len(trajectory) == 0
    assert trajectory.interpolation_method == InterpolationMethod.linear
    assert trajectory.is_empty()


def test_strajectory_with_interpolation_method():
    """Rust test: test_strajectory_with_interpolation_method"""
    # Test creating trajectory with specific interpolation method using builder pattern
    traj = STrajectory6().with_interpolation_method(InterpolationMethod.linear)
    assert traj.interpolation_method == InterpolationMethod.linear
    assert len(traj) == 0

    # Verify it works with adding states
    traj = STrajectory6().with_interpolation_method(InterpolationMethod.linear)
    t0 = Epoch.from_jd(2451545.0, "UTC")
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    traj.add(t0, state)
    assert len(traj) == 1
    assert traj.interpolation_method == InterpolationMethod.linear


def test_strajectory_with_eviction_policy_max_size_builder():
    """Rust test: test_strajectory_with_eviction_policy_max_size_builder"""
    # Test builder pattern for max size eviction policy
    traj = STrajectory6().with_eviction_policy_max_size(5)

    assert traj.get_eviction_policy() == "KeepCount"
    assert len(traj) == 0


def test_strajectory_with_eviction_policy_max_age_builder():
    """Rust test: test_strajectory_with_eviction_policy_max_age_builder"""
    # Test builder pattern for max age eviction policy
    traj = STrajectory6().with_eviction_policy_max_age(300.0)

    assert traj.get_eviction_policy() == "KeepWithinDuration"
    assert len(traj) == 0


def test_strajectory_builder_pattern_chaining():
    """Rust test: test_strajectory_builder_pattern_chaining"""
    # Test chaining multiple builder methods
    traj = (STrajectory6()
            .with_interpolation_method(InterpolationMethod.linear)
            .with_eviction_policy_max_size(10))

    assert traj.interpolation_method == InterpolationMethod.linear
    assert traj.get_eviction_policy() == "KeepCount"

    # Add states and verify eviction policy works
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    for i in range(15):
        epoch = t0 + (i * 60.0)
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    # Should only have 10 states due to eviction policy
    assert len(traj) == 10


def test_strajectory_dimension():
    """Rust test: test_strajectory_dimension"""
    traj = STrajectory6()
    assert traj.dimension() == 6


def test_strajectory_to_matrix():
    """Rust test: test_strajectory_to_matrix"""
    t0 = Epoch.from_jd(2451545.0, "UTC")
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
    ])

    traj = STrajectory6.from_data(epochs, states)

    matrix = traj.to_matrix()

    # Matrix should be 3 rows (time points) x 6 columns (state elements)
    assert matrix.shape == (3, 6)

    # Check first row (first state at t0)
    assert matrix[0, 0] == 1.0
    assert matrix[0, 1] == 2.0
    assert matrix[0, 2] == 3.0
    assert matrix[0, 3] == 4.0
    assert matrix[0, 4] == 5.0
    assert matrix[0, 5] == 6.0

    # Check second row (second state at t1)
    assert matrix[1, 0] == 11.0
    assert matrix[1, 1] == 12.0
    assert matrix[1, 2] == 13.0
    assert matrix[1, 3] == 14.0
    assert matrix[1, 4] == 15.0
    assert matrix[1, 5] == 16.0

    # Check third row (third state at t2)
    assert matrix[2, 0] == 21.0
    assert matrix[2, 1] == 22.0
    assert matrix[2, 2] == 23.0
    assert matrix[2, 3] == 24.0
    assert matrix[2, 4] == 25.0
    assert matrix[2, 5] == 26.0

    # Check first column (first element of each state over time)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 0] == 11.0
    assert matrix[2, 0] == 21.0


# Default Trait Tests

def test_strajectory_default():
    """Rust test: test_strajectory_default"""
    trajectory = STrajectory6()
    assert len(trajectory) == 0
    assert trajectory.interpolation_method == InterpolationMethod.linear
    assert trajectory.is_empty()


# Index Trait Tests

def test_strajectory_index_index():
    """Rust test: test_strajectory_index_index"""
    trajectory = create_test_trajectory()

    # Test indexing returns state vectors
    state0 = trajectory[0]
    assert abs(state0[0] - 7000e3) < 1.0

    state1 = trajectory[1]
    assert abs(state1[0] - 7100e3) < 1.0

    state2 = trajectory[2]
    assert abs(state2[0] - 7200e3) < 1.0


def test_strajectory_index_index_out_of_bounds():
    """Rust test: test_strajectory_index_index_out_of_bounds"""
    trajectory = create_test_trajectory()

    with pytest.raises(IndexError):
        _ = trajectory[10]


# IntoIterator Trait Tests

def test_strajectory_intoiterator_into_iter():
    """Rust test: test_strajectory_intoiterator_into_iter"""
    trajectory = create_test_trajectory()

    count = 0
    for epoch, state in trajectory:
        if count == 0:
            assert epoch.jd() == 2451545.0
            assert abs(state[0] - 7000e3) < 1.0
        elif count == 1:
            assert epoch.jd() == 2451545.1
            assert abs(state[0] - 7100e3) < 1.0
        elif count == 2:
            assert epoch.jd() == 2451545.2
            assert abs(state[0] - 7200e3) < 1.0
        else:
            pytest.fail("Too many iterations")
        count += 1

    assert count == 3


def test_strajectory_intoiterator_into_iter_empty():
    """Rust test: test_strajectory_intoiterator_into_iter_empty"""
    trajectory = STrajectory6()

    count = 0
    for _ in trajectory:
        count += 1

    assert count == 0


def test_strajectory_iterator_iterator_size_hint():
    """Rust test: test_strajectory_iterator_iterator_size_hint

    Note: Python doesn't have a direct equivalent to Rust's size_hint().
    This test verifies the length is correct instead.
    """
    trajectory = create_test_trajectory()

    # In Python, we can check the length directly
    assert len(trajectory) == 3


def test_strajectory_iterator_iterator_len():
    """Rust test: test_strajectory_iterator_iterator_len"""
    trajectory = create_test_trajectory()

    # Verify length
    assert len(trajectory) == 3


# Trajectory Trait Tests

def test_strajectory_trajectory_add():
    """Rust test: test_strajectory_trajectory_add"""
    trajectory = STrajectory6()

    # Add states in order
    epoch1 = Epoch.from_jd(2451545.0, "UTC")
    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    trajectory.add(epoch1, state1)

    epoch3 = Epoch.from_jd(2451545.2, "UTC")
    state3 = np.array([7200e3, 0.0, 0.0, 0.0, 7.7e3, 0.0])
    trajectory.add(epoch3, state3)

    # Add a state in between
    epoch2 = Epoch.from_jd(2451545.1, "UTC")
    state2 = np.array([7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0])
    trajectory.add(epoch2, state2)

    assert len(trajectory) == 3
    assert trajectory.epoch(0).jd() == 2451545.0
    assert trajectory.epoch(1).jd() == 2451545.1
    assert trajectory.epoch(2).jd() == 2451545.2


def test_strajectory_trajectory_state():
    """Rust test: test_strajectory_trajectory_state"""
    trajectory = create_test_trajectory()

    # Test valid indices
    state0 = trajectory.state(0)
    assert state0[0] == 7000e3

    state1 = trajectory.state(1)
    assert state1[0] == 7100e3

    state2 = trajectory.state(2)
    assert state2[0] == 7200e3

    # Test invalid index
    with pytest.raises(Exception):
        trajectory.state(10)


def test_strajectory_trajectory_epoch():
    """Rust test: test_strajectory_trajectory_epoch"""
    trajectory = create_test_trajectory()

    # Test valid indices
    epoch0 = trajectory.epoch(0)
    assert epoch0.jd() == 2451545.0

    epoch1 = trajectory.epoch(1)
    assert epoch1.jd() == 2451545.1

    epoch2 = trajectory.epoch(2)
    assert epoch2.jd() == 2451545.2

    # Test invalid index
    with pytest.raises(Exception):
        trajectory.epoch(10)


def test_strajectory_trajectory_nearest_state():
    """Rust test: test_strajectory_trajectory_nearest_state"""
    trajectory = create_test_trajectory()

    # Request a time exactly at a state
    epoch, state = trajectory.nearest_state(Epoch.from_jd(2451545.0, "UTC"))
    assert epoch.jd() == 2451545.0
    assert state[0] == 7000e3

    # Request a time halfway between two states
    epoch, state = trajectory.nearest_state(Epoch.from_jd(2451545.05, "UTC"))

    # Should return the closest state (first one)
    assert epoch.jd() == 2451545.0
    assert state[0] == 7000e3

    # Request a time right before the second state
    epoch, state = trajectory.nearest_state(Epoch.from_jd(2451545.0999, "UTC"))
    assert epoch.jd() == 2451545.1
    assert state[0] == 7100e3

    # Request a time after the last state
    epoch, state = trajectory.nearest_state(Epoch.from_jd(2451545.3, "UTC"))
    assert epoch.jd() == 2451545.2
    assert state[0] == 7200e3


def test_strajectory_trajectory_len():
    """Rust test: test_strajectory_trajectory_len"""
    trajectory = STrajectory6()
    assert len(trajectory) == 0

    trajectory.add(
        Epoch.from_jd(2451545.0, "UTC"),
        np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    )
    assert len(trajectory) == 1

    trajectory.add(
        Epoch.from_jd(2451545.1, "UTC"),
        np.array([7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0])
    )
    assert len(trajectory) == 2


def test_strajectory_trajectory_is_empty():
    """Rust test: test_strajectory_trajectory_is_empty"""
    trajectory = STrajectory6()
    assert trajectory.is_empty()

    trajectory.add(
        Epoch.from_jd(2451545.0, "UTC"),
        np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    )
    assert not trajectory.is_empty()

    trajectory.clear()
    assert trajectory.is_empty()


def test_strajectory_trajectory_start_epoch():
    """Rust test: test_strajectory_trajectory_start_epoch"""
    trajectory = create_test_trajectory()

    start = trajectory.start_epoch
    assert start.jd() == 2451545.0

    # Test empty trajectory
    empty_trajectory = STrajectory6()
    assert empty_trajectory.start_epoch is None


def test_strajectory_trajectory_end_epoch():
    """Rust test: test_strajectory_trajectory_end_epoch"""
    trajectory = create_test_trajectory()

    end = trajectory.end_epoch
    assert end.jd() == 2451545.2

    # Test empty trajectory
    empty_trajectory = STrajectory6()
    assert empty_trajectory.end_epoch is None


def test_strajectory_trajectory_timespan():
    """Rust test: test_strajectory_trajectory_timespan"""
    trajectory = create_test_trajectory()

    span = trajectory.time_span
    assert abs(span - 0.2 * 86400.0) < 1.0  # 0.2 days in seconds

    # Test single state trajectory
    single_trajectory = STrajectory6()
    single_trajectory.add(
        Epoch.from_jd(2451545.0, "UTC"),
        np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    )
    assert single_trajectory.time_span is None

    # Test empty trajectory
    empty_trajectory = STrajectory6()
    assert empty_trajectory.time_span is None


def test_strajectory_trajectory_first():
    """Rust test: test_strajectory_trajectory_first"""
    # Test empty trajectory
    empty_trajectory = STrajectory6()
    assert empty_trajectory.first() is None

    # Test single state trajectory
    single_trajectory = STrajectory6()
    epoch = Epoch.from_jd(2451545.0, "UTC")
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    single_trajectory.add(epoch, state)

    first_epoch, first_state = single_trajectory.first()
    assert first_epoch.jd() == 2451545.0
    assert first_state[0] == 7000e3

    # Test multi-state trajectory
    trajectory = create_test_trajectory()
    first_epoch, first_state = trajectory.first()
    assert first_epoch.jd() == 2451545.0
    assert first_state[0] == 7000e3


def test_strajectory_trajectory_last():
    """Rust test: test_strajectory_trajectory_last"""
    # Test empty trajectory
    empty_trajectory = STrajectory6()
    assert empty_trajectory.last() is None

    # Test single state trajectory
    single_trajectory = STrajectory6()
    epoch = Epoch.from_jd(2451545.0, "UTC")
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    single_trajectory.add(epoch, state)

    last_epoch, last_state = single_trajectory.last()
    assert last_epoch.jd() == 2451545.0
    assert last_state[0] == 7000e3

    # Test multi-state trajectory
    trajectory = create_test_trajectory()
    last_epoch, last_state = trajectory.last()
    assert last_epoch.jd() == 2451545.2
    assert last_state[0] == 7200e3


def test_strajectory_trajectory_clear():
    """Rust test: test_strajectory_trajectory_clear"""
    trajectory = create_test_trajectory()
    assert len(trajectory) == 3
    assert not trajectory.is_empty()

    trajectory.clear()
    assert len(trajectory) == 0
    assert trajectory.is_empty()
    assert trajectory.start_epoch is None
    assert trajectory.end_epoch is None


def test_strajectory_trajectory_remove_state():
    """Rust test: test_strajectory_trajectory_remove_state"""
    trajectory = create_test_trajectory()

    epoch_to_remove = Epoch.from_jd(2451545.1, "UTC")
    removed_state = trajectory.remove_state(epoch_to_remove)
    assert removed_state[0] == 7100e3
    assert len(trajectory) == 2

    # Test error case
    non_existent_epoch = Epoch.from_jd(2451546.0, "UTC")
    with pytest.raises(Exception):
        trajectory.remove_state(non_existent_epoch)


def test_strajectory_trajectory_remove_state_at_index():
    """Rust test: test_strajectory_trajectory_remove_state_at_index"""
    trajectory = create_test_trajectory()

    removed_epoch, removed_state = trajectory.remove_state_at_index(1)
    assert removed_epoch.jd() == 2451545.1
    assert removed_state[0] == 7100e3
    assert len(trajectory) == 2

    # Test error case
    with pytest.raises(Exception):
        trajectory.remove_state_at_index(10)


def test_strajectory_trajectory_get():
    """Rust test: test_strajectory_trajectory_get"""
    trajectory = create_test_trajectory()

    epoch, state = trajectory.get(0)
    assert epoch.jd() == 2451545.0
    assert state[0] == 7000e3

    # Test bounds checking
    with pytest.raises(Exception):
        trajectory.get(10)


def test_strajectory_trajectory_index_before_epoch():
    """Rust test: test_strajectory_trajectory_index_before_epoch"""
    # Create a trajectory with states at epochs: t0, t0+60s, t0+120s
    t0 = Epoch.from_jd(2451545.0, "UTC")
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    traj = STrajectory6.from_data(epochs, states)

    # Test finding index before t0 (should error - before all states)
    before_t0 = t0 + (-10.0)
    with pytest.raises(Exception):
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

    # Test finding index before t0+150s (should return index 2)
    t0_plus_150 = t0 + 150.0
    idx = traj.index_before_epoch(t0_plus_150)
    assert idx == 2


def test_strajectory_trajectory_index_after_epoch():
    """Rust test: test_strajectory_trajectory_index_after_epoch"""
    # Create a trajectory with states at epochs: t0, t0+60s, t0+120s
    t0 = Epoch.from_jd(2451545.0, "UTC")
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    traj = STrajectory6.from_data(epochs, states)

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

    # Test finding index after t0+120s (should return index 2 - exact match)
    t0_plus_120 = t0 + 120.0
    idx = traj.index_after_epoch(t0_plus_120)
    assert idx == 2

    # Test finding index after t0+150s (should error - after all states)
    t0_plus_150 = t0 + 150.0
    with pytest.raises(Exception):
        traj.index_after_epoch(t0_plus_150)


def test_strajectory_trajectory_state_before_epoch():
    """Rust test: test_strajectory_trajectory_state_before_epoch"""
    # Create a trajectory with distinguishable states at 3 epochs
    t0 = Epoch.from_jd(2451545.0, "UTC")
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array([
        1000.0, 100.0, 10.0, 1.0, 0.1, 0.01,
        2000.0, 200.0, 20.0, 2.0, 0.2, 0.02,
        3000.0, 300.0, 30.0, 3.0, 0.3, 0.03,
    ])
    traj = STrajectory6.from_data(epochs, states)

    # Test error case for epoch before all states
    before_t0 = t0 + (-10.0)
    with pytest.raises(Exception):
        traj.state_before_epoch(before_t0)

    # Test that state_before_epoch returns correct (epoch, state) tuples
    # Test at t0+30s (should return first state)
    t0_plus_30 = t0 + 30.0
    epoch, state = traj.state_before_epoch(t0_plus_30)
    assert epoch == t0
    assert state[0] == 1000.0
    assert state[1] == 100.0

    # Test at exact match t0+60s (should return second state)
    t0_plus_60 = t0 + 60.0
    epoch, state = traj.state_before_epoch(t0_plus_60)
    assert epoch == t0 + 60.0
    assert state[0] == 2000.0
    assert state[1] == 200.0

    # Test at t0+90s (should return second state)
    t0_plus_90 = t0 + 90.0
    epoch, state = traj.state_before_epoch(t0_plus_90)
    assert epoch == t0 + 60.0
    assert state[0] == 2000.0
    assert state[1] == 200.0

    # Test at t0+150s (should return third state)
    t0_plus_150 = t0 + 150.0
    epoch, state = traj.state_before_epoch(t0_plus_150)
    assert epoch == t0 + 120.0
    assert state[0] == 3000.0
    assert state[1] == 300.0

    # Verify it uses the default trait implementation correctly by checking
    # that it produces the same result as calling index_before_epoch + get
    t0_plus_45 = t0 + 45.0
    idx = traj.index_before_epoch(t0_plus_45)
    expected_epoch, expected_state = traj.get(idx)
    actual_epoch, actual_state = traj.state_before_epoch(t0_plus_45)
    assert actual_epoch == expected_epoch
    assert np.array_equal(actual_state, expected_state)


def test_strajectory_trajectory_state_after_epoch():
    """Rust test: test_strajectory_trajectory_state_after_epoch"""
    # Create a trajectory with distinguishable states at 3 epochs
    t0 = Epoch.from_jd(2451545.0, "UTC")
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array([
        1000.0, 100.0, 10.0, 1.0, 0.1, 0.01,
        2000.0, 200.0, 20.0, 2.0, 0.2, 0.02,
        3000.0, 300.0, 30.0, 3.0, 0.3, 0.03,
    ])
    traj = STrajectory6.from_data(epochs, states)

    # Test error case for epoch after all states
    after_t0_120 = t0 + 150.0
    with pytest.raises(Exception):
        traj.state_after_epoch(after_t0_120)

    # Test that state_after_epoch returns correct (epoch, state) tuples
    # Test at t0-30s (should return first state)
    before_t0 = t0 + (-30.0)
    epoch, state = traj.state_after_epoch(before_t0)
    assert epoch == t0
    assert state[0] == 1000.0
    assert state[1] == 100.0

    # Test at exact match t0 (should return first state)
    epoch, state = traj.state_after_epoch(t0)
    assert epoch == t0
    assert state[0] == 1000.0
    assert state[1] == 100.0

    # Test at t0+30s (should return second state)
    t0_plus_30 = t0 + 30.0
    epoch, state = traj.state_after_epoch(t0_plus_30)
    assert epoch == t0 + 60.0
    assert state[0] == 2000.0
    assert state[1] == 200.0

    # Test at exact match t0+60s (should return second state)
    t0_plus_60 = t0 + 60.0
    epoch, state = traj.state_after_epoch(t0_plus_60)
    assert epoch == t0 + 60.0
    assert state[0] == 2000.0
    assert state[1] == 200.0

    # Test at t0+90s (should return third state)
    t0_plus_90 = t0 + 90.0
    epoch, state = traj.state_after_epoch(t0_plus_90)
    assert epoch == t0 + 120.0
    assert state[0] == 3000.0
    assert state[1] == 300.0

    # Verify it uses the default trait implementation correctly by checking
    # that it produces the same result as calling index_after_epoch + get
    t0_plus_45 = t0 + 45.0
    idx = traj.index_after_epoch(t0_plus_45)
    expected_epoch, expected_state = traj.get(idx)
    actual_epoch, actual_state = traj.state_after_epoch(t0_plus_45)
    assert actual_epoch == expected_epoch
    assert np.array_equal(actual_state, expected_state)


def test_strajectory_trajectory_get_eviction_policy():
    """Rust test: test_strajectory_trajectory_get_eviction_policy"""
    traj = STrajectory6()

    # Default is None
    assert traj.get_eviction_policy() == "None"

    # Set to KeepCount
    traj.set_eviction_policy_max_size(10)
    assert traj.get_eviction_policy() == "KeepCount"

    # Set to KeepWithinDuration
    traj.set_eviction_policy_max_age(100.0)
    assert traj.get_eviction_policy() == "KeepWithinDuration"


def test_strajectory_set_eviction_policy_max_size():
    """Rust test: test_strajectory_set_eviction_policy_max_size"""
    traj = STrajectory6()

    # Add 5 states
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    for i in range(5):
        epoch = t0 + (i * 60.0)
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    assert len(traj) == 5

    # Set max size to 3
    traj.set_eviction_policy_max_size(3)

    # Should only have 3 most recent states
    assert len(traj) == 3

    # First state should be the 3rd original state (oldest 2 evicted)
    first_state = traj.state(0)
    assert abs(first_state[0] - (7000e3 + 2000.0)) < 1.0

    # Add another state - should still maintain max size
    new_epoch = t0 + 5.0 * 60.0
    new_state = np.array([7000e3 + 5000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(new_epoch, new_state)

    assert len(traj) == 3
    assert traj.state(0)[0] == 7000e3 + 3000.0

    # Test error case
    with pytest.raises(Exception):
        traj.set_eviction_policy_max_size(0)


def test_strajectory_set_eviction_policy_max_age():
    """Rust test: test_strajectory_set_eviction_policy_max_age"""
    traj = STrajectory6()

    # Add states spanning 5 minutes
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    for i in range(6):
        epoch = t0 + (i * 60.0)  # 0, 60, 120, 180, 240, 300 seconds
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    assert len(traj) == 6

    # Set max age to 240 seconds
    traj.set_eviction_policy_max_age(240.0)
    assert len(traj) == 5
    assert traj.epoch(0) == t0 + 60.0
    assert traj.state(0)[0] == 7000e3 + 1000.0

    # Set max age to 239 seconds
    traj.set_eviction_policy_max_age(239.0)
    assert len(traj) == 4
    assert traj.epoch(0) == t0 + 120.0
    assert traj.state(0)[0] == 7000e3 + 2000.0

    # Test error case
    with pytest.raises(Exception):
        traj.set_eviction_policy_max_age(0.0)
    with pytest.raises(Exception):
        traj.set_eviction_policy_max_age(-10.0)


# Interpolatable Trait Tests

def test_strajectory_interpolatable_get_interpolation_method():
    """Rust test: test_strajectory_interpolatable_get_interpolation_method"""
    traj = STrajectory6()

    # Test default interpolation method is Linear
    assert traj.interpolation_method == InterpolationMethod.linear

    # Test setting to Linear explicitly
    traj.set_interpolation_method(InterpolationMethod.linear)
    assert traj.interpolation_method == InterpolationMethod.linear


def test_strajectory_interpolatable_interpolate_linear():
    """Rust test: test_strajectory_interpolatable_interpolate_linear"""
    t0 = Epoch.from_jd(2451545.0, "UTC")

    # Create a trajectory with 3 states at t0, t0+60s, t0+120s with distinct position values
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7120e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
    ])
    traj = STrajectory6.from_data(epochs, states)

    # Test interpolate_linear at t0+30s (midpoint between first two states)
    # Should be halfway between [7000e3, ...] and [7060e3, ...]
    t_mid = t0 + 30.0
    state_mid = traj.interpolate_linear(t_mid)
    assert abs(state_mid[0] - 7030e3) < 1e-6
    assert abs(state_mid[1] - 0.0) < 1e-6
    assert abs(state_mid[2] - 0.0) < 1e-6
    assert abs(state_mid[3] - 0.0) < 1e-6
    assert abs(state_mid[4] - 7.5e3) < 1e-6
    assert abs(state_mid[5] - 0.0) < 1e-6

    # Test interpolate_linear at exact epochs - should return exact states
    state_t0 = traj.interpolate_linear(t0)
    assert abs(state_t0[0] - 7000e3) < 1e-6

    state_t60 = traj.interpolate_linear(t0 + 60.0)
    assert abs(state_t60[0] - 7060e3) < 1e-6

    state_t120 = traj.interpolate_linear(t0 + 120.0)
    assert abs(state_t120[0] - 7120e3) < 1e-6

    # Test interpolate_linear at t0+90s - should be between second and third states
    # Should be halfway between [7060e3, ...] and [7120e3, ...]
    t_90 = t0 + 90.0
    state_90 = traj.interpolate_linear(t_90)
    assert abs(state_90[0] - 7090e3) < 1e-6
    assert abs(state_90[1] - 0.0) < 1e-6
    assert abs(state_90[2] - 0.0) < 1e-6
    assert abs(state_90[3] - 0.0) < 1e-6
    assert abs(state_90[4] - 7.5e3) < 1e-6
    assert abs(state_90[5] - 0.0) < 1e-6

    # Test error case: single state trajectory (should just return that state)
    single_epochs = [t0]
    single_states = np.array([8000e3, 100.0, 200.0, 1.0, 2.0, 3.0])
    single_traj = STrajectory6.from_data(single_epochs, single_states)

    result = single_traj.interpolate_linear(t0)
    assert abs(result[0] - 8000e3) < 1e-6
    assert abs(result[1] - 100.0) < 1e-6
    assert abs(result[2] - 200.0) < 1e-6
    assert abs(result[3] - 1.0) < 1e-6
    assert abs(result[4] - 2.0) < 1e-6
    assert abs(result[5] - 3.0) < 1e-6

    # Test error case: interpolation on single state trajectory at different epoch
    different_epoch = t0 + 10.0
    with pytest.raises(Exception):
        single_traj.interpolate_linear(different_epoch)

    # Test error case: interpolation on empty trajectory
    empty_traj = STrajectory6()
    with pytest.raises(Exception):
        empty_traj.interpolate_linear(t0)


def test_strajectory_interpolatable_interpolate():
    """Rust test: test_strajectory_interpolatable_interpolate"""
    t0 = Epoch.from_jd(2451545.0, "UTC")

    # Create a trajectory with 3 states at different epochs
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7120e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
    ])
    traj = STrajectory6.from_data(epochs, states)

    # Test that interpolate() returns same result as interpolate_linear() for the same epoch
    t_test = t0 + 30.0
    result_interpolate = traj.interpolate(t_test)
    result_linear = traj.interpolate_linear(t_test)

    assert abs(result_interpolate[0] - result_linear[0]) < 1e-6
    assert abs(result_interpolate[1] - result_linear[1]) < 1e-6
    assert abs(result_interpolate[2] - result_linear[2]) < 1e-6
    assert abs(result_interpolate[3] - result_linear[3]) < 1e-6
    assert abs(result_interpolate[4] - result_linear[4]) < 1e-6
    assert abs(result_interpolate[5] - result_linear[5]) < 1e-6

    # Verify the actual values for completeness
    assert abs(result_interpolate[0] - 7030e3) < 1e-6
