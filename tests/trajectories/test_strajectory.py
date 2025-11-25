"""
Tests for STrajectory6 in brahe.

These tests mirror the Rust STrajectory test suite to ensure 1:1 parity.
Each test corresponds to a specific Rust test in src/trajectories/strajectory.rs.
"""

import pytest
import numpy as np
import brahe
from brahe import Epoch, STrajectory6, InterpolationMethod


def create_test_trajectory():
    """Helper to create a test STrajectory6 with sample data.

    Corresponds to Rust function: create_test_trajectory()
    """
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
        Epoch.from_jd(2451545.2, brahe.UTC),
    ]

    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0],
            [7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0],
        ]
    )

    return STrajectory6.from_data(epochs, states)


# STrajectory Trait Tests


def test_strajectory_strajectory_new():
    """Rust test: test_strajectory_strajectory_new"""
    trajectory = STrajectory6()

    assert len(trajectory) == 0
    assert trajectory.interpolation_method == InterpolationMethod.LINEAR
    assert trajectory.is_empty()


def test_strajectory_with_interpolation_method():
    """Rust test: test_strajectory_with_interpolation_method"""
    # Test creating trajectory with specific interpolation method using builder pattern
    traj = STrajectory6().with_interpolation_method(InterpolationMethod.LINEAR)
    assert traj.interpolation_method == InterpolationMethod.LINEAR
    assert len(traj) == 0

    # Verify it works with adding states
    traj = STrajectory6().with_interpolation_method(InterpolationMethod.LINEAR)
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    traj.add(t0, state)
    assert len(traj) == 1
    assert traj.interpolation_method == InterpolationMethod.LINEAR


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
    traj = (
        STrajectory6()
        .with_interpolation_method(InterpolationMethod.LINEAR)
        .with_eviction_policy_max_size(10)
    )

    assert traj.interpolation_method == InterpolationMethod.LINEAR
    assert traj.get_eviction_policy() == "KeepCount"

    # Add states and verify eviction policy works
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
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
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        ]
    )

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
    assert trajectory.interpolation_method == InterpolationMethod.LINEAR
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
    epoch1 = Epoch.from_jd(2451545.0, brahe.UTC)
    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    trajectory.add(epoch1, state1)

    epoch3 = Epoch.from_jd(2451545.2, brahe.UTC)
    state3 = np.array([7200e3, 0.0, 0.0, 0.0, 7.7e3, 0.0])
    trajectory.add(epoch3, state3)

    # Add a state in between
    epoch2 = Epoch.from_jd(2451545.1, brahe.UTC)
    state2 = np.array([7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0])
    trajectory.add(epoch2, state2)

    assert len(trajectory) == 3
    assert trajectory.epoch_at_idx(0).jd() == 2451545.0
    assert trajectory.epoch_at_idx(1).jd() == 2451545.1
    assert trajectory.epoch_at_idx(2).jd() == 2451545.2


def test_strajectory_trajectory_state():
    """Rust test: test_strajectory_trajectory_state"""
    trajectory = create_test_trajectory()

    # Test valid indices
    state0 = trajectory.state_at_idx(0)
    assert state0[0] == 7000e3

    state1 = trajectory.state_at_idx(1)
    assert state1[0] == 7100e3

    state2 = trajectory.state_at_idx(2)
    assert state2[0] == 7200e3

    # Test invalid index
    with pytest.raises(Exception):
        trajectory.state_at_idx(10)


def test_strajectory_trajectory_epoch():
    """Rust test: test_strajectory_trajectory_epoch"""
    trajectory = create_test_trajectory()

    # Test valid indices
    epoch0 = trajectory.epoch_at_idx(0)
    assert epoch0.jd() == 2451545.0

    epoch1 = trajectory.epoch_at_idx(1)
    assert epoch1.jd() == 2451545.1

    epoch2 = trajectory.epoch_at_idx(2)
    assert epoch2.jd() == 2451545.2

    # Test invalid index
    with pytest.raises(Exception):
        trajectory.epoch_at_idx(10)


def test_strajectory_trajectory_nearest_state():
    """Rust test: test_strajectory_trajectory_nearest_state"""
    trajectory = create_test_trajectory()

    # Request a time exactly at a state
    epoch, state = trajectory.nearest_state(Epoch.from_jd(2451545.0, brahe.UTC))
    assert epoch.jd() == 2451545.0
    assert state[0] == 7000e3

    # Request a time halfway between two states
    epoch, state = trajectory.nearest_state(Epoch.from_jd(2451545.05, brahe.UTC))

    # Should return the closest state (first one)
    assert epoch.jd() == 2451545.0
    assert state[0] == 7000e3

    # Request a time right before the second state
    epoch, state = trajectory.nearest_state(Epoch.from_jd(2451545.0999, brahe.UTC))
    assert epoch.jd() == 2451545.1
    assert state[0] == 7100e3

    # Request a time after the last state
    epoch, state = trajectory.nearest_state(Epoch.from_jd(2451545.3, brahe.UTC))
    assert epoch.jd() == 2451545.2
    assert state[0] == 7200e3


def test_strajectory_trajectory_len():
    """Rust test: test_strajectory_trajectory_len"""
    trajectory = STrajectory6()
    assert len(trajectory) == 0

    trajectory.add(
        Epoch.from_jd(2451545.0, brahe.UTC),
        np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
    )
    assert len(trajectory) == 1

    trajectory.add(
        Epoch.from_jd(2451545.1, brahe.UTC),
        np.array([7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0]),
    )
    assert len(trajectory) == 2


def test_strajectory_trajectory_is_empty():
    """Rust test: test_strajectory_trajectory_is_empty"""
    trajectory = STrajectory6()
    assert trajectory.is_empty()

    trajectory.add(
        Epoch.from_jd(2451545.0, brahe.UTC),
        np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
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
        Epoch.from_jd(2451545.0, brahe.UTC),
        np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
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
    epoch = Epoch.from_jd(2451545.0, brahe.UTC)
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
    epoch = Epoch.from_jd(2451545.0, brahe.UTC)
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


def test_strajectory_trajectory_remove_epoch():
    """Rust test: test_strajectory_trajectory_remove_epoch"""
    trajectory = create_test_trajectory()

    epoch_to_remove = Epoch.from_jd(2451545.1, brahe.UTC)
    removed_state = trajectory.remove_epoch(epoch_to_remove)
    assert removed_state[0] == 7100e3
    assert len(trajectory) == 2

    # Test error case
    non_existent_epoch = Epoch.from_jd(2451546.0, brahe.UTC)
    with pytest.raises(Exception):
        trajectory.remove_epoch(non_existent_epoch)


def test_strajectory_trajectory_remove():
    """Rust test: test_strajectory_trajectory_remove"""
    trajectory = create_test_trajectory()

    removed_epoch, removed_state = trajectory.remove(1)
    assert removed_epoch.jd() == 2451545.1
    assert removed_state[0] == 7100e3
    assert len(trajectory) == 2

    # Test error case
    with pytest.raises(Exception):
        trajectory.remove(10)


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
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
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
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
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
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array(
        [
            [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01],
            [2000.0, 200.0, 20.0, 2.0, 0.2, 0.02],
            [3000.0, 300.0, 30.0, 3.0, 0.3, 0.03],
        ]
    )
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
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array(
        [
            [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01],
            [2000.0, 200.0, 20.0, 2.0, 0.2, 0.02],
            [3000.0, 300.0, 30.0, 3.0, 0.3, 0.03],
        ]
    )
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
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
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
    first_state = traj.state_at_idx(0)
    assert abs(first_state[0] - (7000e3 + 2000.0)) < 1.0

    # Add another state - should still maintain max size
    new_epoch = t0 + 5.0 * 60.0
    new_state = np.array([7000e3 + 5000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(new_epoch, new_state)

    assert len(traj) == 3
    assert traj.state_at_idx(0)[0] == 7000e3 + 3000.0

    # Test error case
    with pytest.raises(Exception):
        traj.set_eviction_policy_max_size(0)


def test_strajectory_set_eviction_policy_max_age():
    """Rust test: test_strajectory_set_eviction_policy_max_age"""
    traj = STrajectory6()

    # Add states spanning 5 minutes
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    for i in range(6):
        epoch = t0 + (i * 60.0)  # 0, 60, 120, 180, 240, 300 seconds
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    assert len(traj) == 6

    # Set max age to 240 seconds
    traj.set_eviction_policy_max_age(240.0)
    assert len(traj) == 5
    assert traj.epoch_at_idx(0) == t0 + 60.0
    assert traj.state_at_idx(0)[0] == 7000e3 + 1000.0

    # Set max age to 239 seconds
    traj.set_eviction_policy_max_age(239.0)
    assert len(traj) == 4
    assert traj.epoch_at_idx(0) == t0 + 120.0
    assert traj.state_at_idx(0)[0] == 7000e3 + 2000.0

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
    assert traj.interpolation_method == InterpolationMethod.LINEAR

    # Test setting to Linear explicitly
    traj.set_interpolation_method(InterpolationMethod.LINEAR)
    assert traj.interpolation_method == InterpolationMethod.LINEAR


def test_strajectory_interpolatable_interpolate_linear():
    """Rust test: test_strajectory_interpolatable_interpolate_linear"""
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)

    # Create a trajectory with 3 states at t0, t0+60s, t0+120s with distinct position values
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7120e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
        ]
    )
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
    single_states = np.array([[8000e3, 100.0, 200.0, 1.0, 2.0, 3.0]])
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
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)

    # Create a trajectory with 3 states at different epochs
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7120e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
        ]
    )
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


def test_strajectory_interpolate_before_start():
    """Rust test: test_strajectory_interpolate_before_start"""
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)

    # Create a trajectory with states from t0 to t0+120s
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7120e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
        ]
    )
    traj = STrajectory6.from_data(epochs, states)

    # Test interpolation before trajectory start
    before_start = t0 - 10.0
    with pytest.raises(Exception):
        traj.interpolate_linear(before_start)

    # Also test with interpolate() method
    with pytest.raises(Exception):
        traj.interpolate(before_start)


def test_strajectory_interpolate_after_end():
    """Rust test: test_strajectory_interpolate_after_end"""
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)

    # Create a trajectory with states from t0 to t0+120s
    epochs = [
        t0,
        t0 + 60.0,
        t0 + 120.0,
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7120e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
        ]
    )
    traj = STrajectory6.from_data(epochs, states)

    # Test interpolation after trajectory end
    after_end = t0 + 130.0
    with pytest.raises(Exception):
        traj.interpolate_linear(after_end)

    # Also test with interpolate() method
    with pytest.raises(Exception):
        traj.interpolate(after_end)


def test_strajectory_interpolate_empty_trajectory():
    """Rust test: test_strajectory_interpolate_empty_trajectory"""
    # Test that interpolating from an empty trajectory returns an error
    traj = STrajectory6()
    t = Epoch.from_jd(2451545.0, brahe.UTC)

    # Test with interpolate_linear
    with pytest.raises(Exception) as exc_info:
        traj.interpolate_linear(t)
    assert "empty trajectory" in str(exc_info.value).lower()

    # Test with interpolate
    with pytest.raises(Exception):
        traj.interpolate(t)


def test_strajectory_interpolate_single_state_exact_match():
    """Rust test: test_strajectory_interpolate_single_state_exact_match"""
    # Test that interpolating at exact epoch in single-state trajectory returns the state
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    state0 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    traj = STrajectory6()
    traj.add(t0, state0)

    # Test interpolation at exact epoch - should return the state
    result = traj.interpolate_linear(t0)
    assert result[0] == pytest.approx(state0[0], abs=1e-6)
    assert result[1] == pytest.approx(state0[1], abs=1e-6)
    assert result[2] == pytest.approx(state0[2], abs=1e-6)

    # Also test with interpolate() method
    result = traj.interpolate(t0)
    assert result[0] == pytest.approx(state0[0], abs=1e-6)


def test_strajectory_interpolate_single_state_no_match():
    """Rust test: test_strajectory_interpolate_single_state_no_match"""
    # Test that interpolating at different epoch in single-state trajectory returns error
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    state0 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    traj = STrajectory6()
    traj.add(t0, state0)

    # Test interpolation at different epoch - should error
    t_different = t0 + 60.0
    with pytest.raises(Exception) as exc_info:
        traj.interpolate_linear(t_different)
    assert "single state" in str(exc_info.value).lower()

    # Also test with interpolate() method
    with pytest.raises(Exception):
        traj.interpolate(t_different)


# Covariance Storage Tests


def test_strajectory6_enable_covariance_storage():
    """Test enabling covariance storage on a trajectory"""
    traj = STrajectory6()

    # Enable covariance storage
    traj.enable_covariance_storage()

    # Add a state - covariance should default to zeros
    t0 = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    traj.add(t0, state)

    # Should be able to query covariance (will be zeros)
    cov = traj.covariance_at(t0)
    assert cov is not None
    assert cov.shape == (6, 6)
    assert np.allclose(cov, np.zeros((6, 6)))


def test_strajectory6_add_with_covariance():
    """Test adding states with covariance matrices"""
    traj = STrajectory6()

    # Add states with covariances
    t0 = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state0 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    cov0 = np.eye(6) * 100.0  # 100 m²/m²/s² diagonal covariance
    traj.add_with_covariance(t0, state0, cov0)

    t1 = t0 + 60.0
    state1 = np.array([brahe.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7550.0, 0.0])
    cov1 = np.eye(6) * 200.0
    traj.add_with_covariance(t1, state1, cov1)

    # Verify covariances are stored
    retrieved_cov0 = traj.covariance_at(t0)
    assert retrieved_cov0 is not None
    assert np.allclose(retrieved_cov0, cov0)

    retrieved_cov1 = traj.covariance_at(t1)
    assert retrieved_cov1 is not None
    assert np.allclose(retrieved_cov1, cov1)


def test_strajectory6_set_covariance_at():
    """Test setting covariance at a specific index"""
    traj = STrajectory6()

    # Add state without covariance first
    t0 = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    traj.add(t0, state)

    # Set covariance at index 0
    cov = np.eye(6) * 150.0
    traj.set_covariance_at(0, cov)

    # Verify covariance is set
    retrieved_cov = traj.covariance_at(t0)
    assert retrieved_cov is not None
    assert np.allclose(retrieved_cov, cov)


def test_strajectory6_covariance_interpolation():
    """Test covariance interpolation at intermediate epochs"""
    traj = STrajectory6()

    # Add states with covariances at t0 and t2
    t0 = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state0 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    cov0 = np.eye(6) * 100.0
    traj.add_with_covariance(t0, state0, cov0)

    t2 = t0 + 120.0
    state2 = np.array([brahe.R_EARTH + 520e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    cov2 = np.eye(6) * 200.0
    traj.add_with_covariance(t2, state2, cov2)

    # Query covariance at midpoint (should be interpolated)
    t1 = t0 + 60.0
    cov_interp = traj.covariance_at(t1)
    assert cov_interp is not None
    assert cov_interp.shape == (6, 6)

    # Diagonal should be approximately halfway between 100 and 200
    # (linear interpolation in covariance space)
    for i in range(6):
        assert cov_interp[i, i] > 100.0
        assert cov_interp[i, i] < 200.0


def test_strajectory6_covariance_without_initialization_returns_none():
    """Test that covariance returns None when not initialized"""
    traj = STrajectory6()

    # Add state without covariance
    t0 = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    traj.add(t0, state)

    # Query covariance should return None
    cov = traj.covariance_at(t0)
    assert cov is None


# Covariance Interpolation Configuration Tests


def test_strajectory6_covariance_interpolation_config():
    """Rust test: test_strajectory_covariance_interpolation_config

    Test the CovarianceInterpolationConfig trait implementation.
    """
    # Test default is TwoWasserstein
    traj = STrajectory6()
    assert (
        traj.get_covariance_interpolation_method()
        == brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )

    # Test with_covariance_interpolation_method builder
    traj = STrajectory6().with_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )
    assert (
        traj.get_covariance_interpolation_method()
        == brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )

    # Test set_covariance_interpolation_method
    traj = STrajectory6()
    traj.set_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )
    assert (
        traj.get_covariance_interpolation_method()
        == brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )
    traj.set_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )
    assert (
        traj.get_covariance_interpolation_method()
        == brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )


def test_strajectory6_covariance_interpolation_methods():
    """Rust test: test_strajectory_covariance_interpolation_methods

    Test that covariance interpolation produces correct results.
    """
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    t1 = t0 + 60.0

    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0])

    # Create diagonal covariance matrices
    cov1 = np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
    cov2 = np.diag([200.0, 200.0, 200.0, 2.0, 2.0, 2.0])

    # Create trajectory with covariances
    traj = STrajectory6()
    traj.enable_covariance_storage()
    traj.add(t0, state1)
    traj.add(t1, state2)
    traj.set_covariance_at(0, cov1)
    traj.set_covariance_at(1, cov2)

    # Test matrix square root interpolation at midpoint
    traj.set_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )
    t_mid = t0 + 30.0
    cov_sqrt = traj.covariance_at(t_mid)

    # Check symmetry and positive semi-definiteness
    for i in range(6):
        assert cov_sqrt[i, i] > 0.0
        for j in range(6):
            assert cov_sqrt[i, j] == pytest.approx(cov_sqrt[j, i], abs=1e-10)

    # Check values are between endpoints
    assert cov_sqrt[0, 0] > 100.0 and cov_sqrt[0, 0] < 200.0

    # Test two-Wasserstein interpolation at midpoint
    traj.set_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )
    cov_wasserstein = traj.covariance_at(t_mid)

    # Check symmetry and positive semi-definiteness
    for i in range(6):
        assert cov_wasserstein[i, i] > 0.0
        for j in range(6):
            assert cov_wasserstein[i, j] == pytest.approx(
                cov_wasserstein[j, i], abs=1e-10
            )

    # Check values are between endpoints
    assert cov_wasserstein[0, 0] > 100.0 and cov_wasserstein[0, 0] < 200.0

    # For diagonal matrices, both methods should give similar results
    assert cov_sqrt[0, 0] == pytest.approx(cov_wasserstein[0, 0], abs=1e-6)


def test_strajectory6_covariance_at_exact_epochs():
    """Rust test: test_strajectory_covariance_at_exact_epochs

    Test that covariance_at returns exact values at data points.
    """
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    t1 = t0 + 60.0

    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0])

    cov1 = np.eye(6) * 100.0
    cov2 = np.eye(6) * 200.0

    traj = STrajectory6()
    traj.enable_covariance_storage()
    traj.add(t0, state1)
    traj.add(t1, state2)
    traj.set_covariance_at(0, cov1)
    traj.set_covariance_at(1, cov2)

    # At exact t0, should return cov1
    result = traj.covariance_at(t0)
    assert result[0, 0] == pytest.approx(100.0, abs=1e-10)

    # At exact t1, should return cov2
    result = traj.covariance_at(t1)
    assert result[0, 0] == pytest.approx(200.0, abs=1e-10)
