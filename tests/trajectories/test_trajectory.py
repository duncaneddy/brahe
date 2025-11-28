"""Tests for Trajectory in brahe - 1:1 parity with Rust tests"""

import pytest
import numpy as np
import brahe
from brahe import Epoch, Trajectory, InterpolationMethod


def create_test_trajectory():
    """Helper function matching Rust create_test_trajectory()"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
        Epoch.from_jd(2451545.2, brahe.UTC),
    ]

    states = [
        np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
        np.array([7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0]),
        np.array([7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0]),
    ]

    traj = Trajectory(6)
    for epoch, state in zip(epochs, states):
        traj.add(epoch, state)

    return traj


# Trajectory Trait Tests


def test_trajectory_new_with_dimension():
    """Rust: test_trajectory_new_with_dimension"""
    # 3
    traj = Trajectory(3)
    assert traj.dimension() == 3
    assert len(traj) == 0
    assert traj.is_empty()

    # 6
    traj = Trajectory(6)
    assert traj.dimension() == 6
    assert len(traj) == 0
    assert traj.is_empty()

    # 12
    traj = Trajectory(12)
    assert traj.dimension() == 12
    assert len(traj) == 0
    assert traj.is_empty()


def test_trajectory_new_with_zero_dimension():
    """Rust: test_trajectory_new_with_zero_dimension"""
    with pytest.raises(Exception, match="Trajectory dimension must be greater than 0"):
        Trajectory(0)


def test_trajectory_with_interpolation_method():
    """Rust: test_trajectory_with_interpolation_method"""
    traj = Trajectory(12).with_interpolation_method(InterpolationMethod.LINEAR)
    assert traj.dimension() == 12
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR


def test_trajectory_with_eviction_policy_max_size_builder():
    """Rust: test_trajectory_with_eviction_policy_max_size_builder"""
    # Test builder pattern for max size eviction policy
    traj = Trajectory(6).with_eviction_policy_max_size(5)

    assert traj.get_eviction_policy() == "KeepCount"
    assert len(traj) == 0


def test_trajectory_with_eviction_policy_max_age_builder():
    """Rust: test_trajectory_with_eviction_policy_max_age_builder"""
    # Test builder pattern for max age eviction policy
    traj = Trajectory(6).with_eviction_policy_max_age(300.0)

    assert traj.get_eviction_policy() == "KeepWithinDuration"
    assert len(traj) == 0


def test_trajectory_builder_pattern_chaining():
    """Rust: test_trajectory_builder_pattern_chaining"""
    # Test chaining multiple builder methods
    traj = (
        Trajectory(6)
        .with_interpolation_method(InterpolationMethod.LINEAR)
        .with_eviction_policy_max_size(10)
    )

    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR
    assert traj.get_eviction_policy() == "KeepCount"

    # Add states and verify eviction policy works
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    for i in range(15):
        epoch = t0 + (i * 60.0)
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    # Should only have 10 states due to eviction policy
    assert len(traj) == 10


def test_trajectory_dimension():
    """Rust: test_trajectory_dimension"""
    traj = Trajectory(9)
    assert traj.dimension() == 9

    traj = Trajectory(4)
    assert traj.dimension() == 4


def test_trajectory_interpolatable_set_interpolation_method():
    """Rust: test_trajectory_interpolatable_set_interpolation_method"""
    traj = Trajectory(6)
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR

    traj.set_interpolation_method(InterpolationMethod.LINEAR)
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR


def test_trajectory_to_matrix():
    """Rust: test_trajectory_to_matrix"""
    traj = create_test_trajectory()
    matrix = traj.to_matrix()

    # Matrix should be 3 rows (time points) x 6 columns (state elements)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == 6

    # Test first row (first state at t0)
    assert matrix[0, 0] == pytest.approx(7000e3, abs=1.0)
    assert matrix[0, 1] == pytest.approx(0.0, abs=1.0)
    assert matrix[0, 2] == pytest.approx(0.0, abs=1.0)
    assert matrix[0, 3] == pytest.approx(0.0, abs=1.0)
    assert matrix[0, 4] == pytest.approx(7.5e3, abs=1.0)
    assert matrix[0, 5] == pytest.approx(0.0, abs=1.0)

    # Test second row (second state at t1)
    assert matrix[1, 0] == pytest.approx(7100e3, abs=1.0)
    assert matrix[1, 1] == pytest.approx(1000e3, abs=1.0)

    # Test third row (third state at t2)
    assert matrix[2, 0] == pytest.approx(7200e3, abs=1.0)
    assert matrix[2, 1] == pytest.approx(2000e3, abs=1.0)
    assert matrix[2, 2] == pytest.approx(1000e3, abs=1.0)
    assert matrix[2, 3] == pytest.approx(200.0, abs=1.0)
    assert matrix[2, 4] == pytest.approx(7.7e3, abs=1.0)
    assert matrix[2, 5] == pytest.approx(100.0, abs=1.0)

    # Test first column (first element of each state over time)
    assert matrix[0, 0] == pytest.approx(7000e3, abs=1.0)
    assert matrix[1, 0] == pytest.approx(7100e3, abs=1.0)
    assert matrix[2, 0] == pytest.approx(7200e3, abs=1.0)


def test_trajectory_trajectory_get_eviction_policy():
    """Rust: test_trajectory_trajectory_get_eviction_policy"""
    traj = Trajectory(6)

    # Default is None
    assert traj.get_eviction_policy() == "None"

    # Set to KeepCount
    traj.set_eviction_policy_max_size(10)
    assert traj.get_eviction_policy() == "KeepCount"

    # Set to KeepWithinDuration
    traj.set_eviction_policy_max_age(100.0)
    assert traj.get_eviction_policy() == "KeepWithinDuration"


def test_trajectory_apply_eviction_policy_keep_count():
    """Rust: test_trajectory_apply_eviction_policy_keep_count"""
    traj = Trajectory(6).with_eviction_policy_max_size(3)

    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    for i in range(5):
        epoch = t0 + (i * 60.0)
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    # Should only have 3 states due to eviction policy
    assert len(traj) == 3
    assert (
        traj.epoch_at_idx(0) == t0 + 2.0 * 60.0
    )  # First state should be the third added


def test_trajectory_apply_eviction_policy_keep_within_duration():
    """Rust: test_trajectory_apply_eviction_policy_keep_within_duration"""
    traj = Trajectory(6).with_eviction_policy_max_age(86400.0 * 7.0 - 1.0)  # 7 days

    t0 = Epoch.from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    for i in range(10):
        epoch = t0 + (i * 86400.0)  # 1 day apart
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    # Should only have 7 states due to eviction policy
    assert len(traj) == 7
    assert (
        traj.epoch_at_idx(0) == t0 + 3.0 * 86400.0
    )  # First state should be the fourth added

    # Repeat with an exact 7 days limit
    traj = Trajectory(6).with_eviction_policy_max_age(86400.0 * 7.0)  # 7 days
    for i in range(10):
        epoch = t0 + (i * 86400.0)
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    # Should still have 8 states due to exact 7 days limit
    assert len(traj) == 8
    assert (
        traj.epoch_at_idx(0) == t0 + 2.0 * 86400.0
    )  # First state should be the third added


# Default Trait Tests


def test_trajectory_default():
    """Rust: test_trajectory_default"""
    traj = Trajectory()
    assert traj.dimension() == 6
    assert len(traj) == 0
    assert traj.is_empty()
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR
    assert traj.get_eviction_policy() == "None"


# Index Trait Tests


def test_trajectory_index():
    """Rust: test_trajectory_index"""
    traj = create_test_trajectory()
    state = traj[0]

    assert len(state) == 6
    assert state[0] == 7000e3
    assert state[1] == 0.0
    assert state[2] == 0.0
    assert state[3] == 0.0
    assert state[4] == 7.5e3
    assert state[5] == 0.0

    state = traj[1]
    assert state[0] == 7100e3
    assert state[1] == 1000e3
    assert state[2] == 500e3
    assert state[3] == 100.0
    assert state[4] == 7.6e3
    assert state[5] == 50.0

    state = traj[2]
    assert state[0] == 7200e3
    assert state[1] == 2000e3
    assert state[2] == 1000e3
    assert state[3] == 200.0
    assert state[4] == 7.7e3
    assert state[5] == 100.0


def test_trajectory_index_index_out_of_bounds():
    """Rust: test_trajectory_index_index_out_of_bounds"""
    traj = create_test_trajectory()
    with pytest.raises(IndexError):
        _ = traj[10]


# IntoIterator Trait Tests


def test_trajectory_intoiterator_into_iter():
    """Rust: test_trajectory_intoiterator_into_iter"""
    traj = create_test_trajectory()

    count = 0
    for epoch, state in traj:
        if count == 0:
            assert epoch.jd() == 2451545.0
            assert state[0] == pytest.approx(7000e3, abs=1.0)
        elif count == 1:
            assert epoch.jd() == 2451545.1
            assert state[0] == pytest.approx(7100e3, abs=1.0)
        elif count == 2:
            assert epoch.jd() == 2451545.2
            assert state[0] == pytest.approx(7200e3, abs=1.0)
        else:
            pytest.fail("Too many iterations")
        count += 1
    assert count == 3


def test_trajectory_intoiterator_into_iter_empty():
    """Rust: test_trajectory_intoiterator_into_iter_empty"""
    traj = Trajectory(6)

    count = 0
    for _ in traj:
        count += 1
    assert count == 0


# Trajectory Trait Tests


def test_trajectory_from_data():
    """Rust: test_trajectory_from_data"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    # States as 2D array: shape (num_epochs, dimension) = (2, 3)
    states = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    traj = Trajectory.from_data(epochs, states)
    assert traj.dimension() == 3
    assert len(traj) == 2


def test_trajectory_from_data_errors():
    """Rust: test_trajectory_from_data_errors"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    # Mismatched: 2 epochs but only 1 state
    states = np.array(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(Exception):
        Trajectory.from_data(epochs, states)

    empty_epochs = []
    empty_states = np.array([]).reshape(0, 3)  # Empty 2D array
    with pytest.raises(Exception):
        Trajectory.from_data(empty_epochs, empty_states)


def test_trajectory_trajectory_add():
    """Rust: test_trajectory_trajectory_add"""
    trajectory = Trajectory(6)

    epoch1 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    trajectory.add(epoch1, state1)
    assert len(trajectory) == 1

    epoch2 = Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, brahe.UTC)
    state2 = np.array([7100e3, 100e3, 50e3, 10.0, 7.6e3, 5.0])

    trajectory.add(epoch2, state2)
    assert len(trajectory) == 2

    np.testing.assert_array_equal(trajectory.state_at_idx(0), state1)
    np.testing.assert_array_equal(trajectory.state_at_idx(1), state2)


def test_trajectory_trajectory_add_out_of_order():
    """Rust: test_trajectory_trajectory_add_out_of_order"""
    trajectory = Trajectory(6)
    epoch1 = Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([7100e3, 100e3, 60e3, 10.0, 7.6e3, 5.0])

    trajectory.add(epoch1, state1)
    assert len(trajectory) == 1
    assert trajectory.epoch_at_idx(0) == epoch1
    np.testing.assert_array_equal(trajectory.state_at_idx(0), state1)

    epoch2 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state2 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    trajectory.add(epoch2, state2)
    assert len(trajectory) == 2
    assert trajectory.epoch_at_idx(0) == epoch2
    np.testing.assert_array_equal(trajectory.state_at_idx(0), state2)
    assert trajectory.epoch_at_idx(1) == epoch1
    np.testing.assert_array_equal(trajectory.state_at_idx(1), state1)


def test_trajectory_trajectory_add_dimension_mismatch():
    """Rust: test_trajectory_trajectory_add_dimension_mismatch"""
    trajectory = Trajectory(6)
    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([7000e3, 0.0, 0.0])  # Dimension 3 instead of 6

    with pytest.raises(Exception):
        trajectory.add(epoch, state)


def test_trajectory_trajectory_add_same_time():
    """Rust: test_trajectory_trajectory_add_append"""
    trajectory = Trajectory(6)
    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    trajectory.add(epoch, state1)
    assert len(trajectory) == 1
    np.testing.assert_array_equal(trajectory.state_at_idx(0), state1)

    state2 = np.array([7100e3, 100e3, 50e3, 10.0, 7.6e3, 5.0])
    trajectory.add(epoch, state2)
    assert len(trajectory) == 2  # Length should increment by one (append behavior)
    # Append behavior: both states preserved, state1 at index 0, state2 at index 1
    np.testing.assert_array_equal(
        trajectory.state_at_idx(0), state1
    )  # First state unchanged
    np.testing.assert_array_equal(
        trajectory.state_at_idx(1), state2
    )  # Second state appended


def test_trajectory_trajectory_epoch():
    """Rust: test_trajectory_trajectory_epoch"""
    traj = create_test_trajectory()

    epoch = traj.epoch_at_idx(0)
    assert epoch == Epoch.from_jd(2451545.0, brahe.UTC)

    epoch = traj.epoch_at_idx(1)
    assert epoch == Epoch.from_jd(2451545.1, brahe.UTC)


def test_trajectory_trajectory_state():
    """Rust: test_trajectory_trajectory_state"""
    traj = create_test_trajectory()

    state = traj.state_at_idx(0)
    assert state[0] == pytest.approx(7000e3, abs=1.0)

    state = traj.state_at_idx(1)
    assert state[0] == pytest.approx(7100e3, abs=1.0)


def test_trajectory_trajectory_nearest_state():
    """Rust: test_trajectory_trajectory_nearest_state"""
    traj = create_test_trajectory()

    # Halfway between first and second
    epoch = Epoch.from_jd(2451545.05, brahe.UTC)
    nearest_epoch, _ = traj.nearest_state(epoch)
    assert nearest_epoch == Epoch.from_jd(2451545.0, brahe.UTC)

    # Slightly before the second
    epoch = Epoch.from_jd(2451545.09, brahe.UTC)
    nearest_epoch, _ = traj.nearest_state(epoch)
    assert nearest_epoch == Epoch.from_jd(2451545.1, brahe.UTC)

    # Slightly after the second
    epoch = Epoch.from_jd(2451545.11, brahe.UTC)
    nearest_epoch, _ = traj.nearest_state(epoch)
    assert nearest_epoch == Epoch.from_jd(2451545.1, brahe.UTC)

    # Exactly at the third
    epoch = Epoch.from_jd(2451545.2, brahe.UTC)
    nearest_epoch, _ = traj.nearest_state(epoch)
    assert nearest_epoch == Epoch.from_jd(2451545.2, brahe.UTC)


def test_trajectory_trajectory_len():
    """Rust: test_trajectory_trajectory_len"""
    traj = create_test_trajectory()
    assert len(traj) == 3

    empty_traj = Trajectory(6)
    assert len(empty_traj) == 0


def test_trajectory_trajectory_is_empty():
    """Rust: test_trajectory_trajectory_is_empty"""
    traj = create_test_trajectory()
    assert not traj.is_empty()

    empty_traj = Trajectory(6)
    assert empty_traj.is_empty()


def test_trajectory_trajectory_start_epoch():
    """Rust: test_trajectory_trajectory_start_epoch"""
    traj = create_test_trajectory()
    start = traj.start_epoch()
    assert start == Epoch.from_jd(2451545.0, brahe.UTC)

    empty_traj = Trajectory(6)
    assert empty_traj.start_epoch() is None


def test_trajectory_trajectory_end_epoch():
    """Rust: test_trajectory_trajectory_end_epoch"""
    traj = create_test_trajectory()
    end = traj.end_epoch()
    assert end == Epoch.from_jd(2451545.2, brahe.UTC)

    empty_traj = Trajectory(6)
    assert empty_traj.end_epoch() is None


def test_trajectory_trajectory_timespan():
    """Rust: test_trajectory_trajectory_timespan"""
    traj = create_test_trajectory()
    timespan = traj.timespan()
    assert timespan == pytest.approx(0.2 * 86400.0, abs=1.0)

    empty_traj = Trajectory(6)
    assert empty_traj.timespan() is None


def test_trajectory_trajectory_first():
    """Rust: test_trajectory_trajectory_first"""
    traj = create_test_trajectory()
    epoch, state = traj.first()
    assert epoch == Epoch.from_jd(2451545.0, brahe.UTC)
    assert state[0] == pytest.approx(7000e3, abs=1.0)

    empty_traj = Trajectory(6)
    assert empty_traj.first() is None


def test_trajectory_trajectory_last():
    """Rust: test_trajectory_trajectory_last"""
    traj = create_test_trajectory()
    epoch, state = traj.last()
    assert epoch == Epoch.from_jd(2451545.2, brahe.UTC)
    assert state[0] == pytest.approx(7200e3, abs=1.0)

    empty_traj = Trajectory(6)
    assert empty_traj.last() is None


def test_trajectory_trajectory_clear():
    """Rust: test_trajectory_trajectory_clear"""
    traj = create_test_trajectory()
    assert len(traj) == 3

    traj.clear()
    assert len(traj) == 0
    assert traj.is_empty()


def test_trajectory_trajectory_remove_epoch():
    """Rust: test_trajectory_trajectory_remove_epoch"""
    traj = create_test_trajectory()
    epoch = Epoch.from_jd(2451545.1, brahe.UTC)

    removed_state = traj.remove_epoch(epoch)
    assert removed_state[0] == pytest.approx(7100e3, abs=1.0)
    assert len(traj) == 2


def test_trajectory_trajectory_remove():
    """Rust: test_trajectory_trajectory_remove"""
    traj = create_test_trajectory()

    removed_epoch, removed_state = traj.remove(1)
    assert removed_epoch == Epoch.from_jd(2451545.1, brahe.UTC)
    assert removed_state[0] == pytest.approx(7100e3, abs=1.0)
    assert len(traj) == 2


def test_trajectory_trajectory_remove_out_of_bounds():
    """Rust: test_trajectory_trajectory_remove_out_of_bounds"""
    traj = create_test_trajectory()

    with pytest.raises(Exception):
        traj.remove(10)


def test_trajectory_trajectory_get():
    """Rust: test_trajectory_trajectory_get"""
    traj = create_test_trajectory()

    epoch, state = traj.get(1)
    assert epoch == Epoch.from_jd(2451545.1, brahe.UTC)
    assert state[0] == pytest.approx(7100e3, abs=1.0)


def test_trajectory_trajectory_index_before_epoch():
    """Rust: test_trajectory_trajectory_index_before_epoch"""
    # Create a 6-dimensional Trajectory with states at epochs: t0, t0+60s, t0+120s
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        ]
    )

    traj = Trajectory.from_data(epochs, states)

    # Test finding index before t0 (should error - before all states)
    before_t0 = t0 + (-10.0)
    with pytest.raises(Exception):
        traj.index_before_epoch(before_t0)

    # Test finding index before t0+30s (should return index 0)
    t0_plus_30 = t0 + 30.0
    assert traj.index_before_epoch(t0_plus_30) == 0

    # Test finding index before t0+60s (should return index 1 - exact match)
    assert traj.index_before_epoch(t1) == 1

    # Test finding index before t0+90s (should return index 1)
    t0_plus_90 = t0 + 90.0
    assert traj.index_before_epoch(t0_plus_90) == 1

    # Test finding index before t0+120s (should return index 2 - exact match)
    assert traj.index_before_epoch(t2) == 2

    # Test finding index before t0+150s (should return index 2)
    t0_plus_150 = t0 + 150.0
    assert traj.index_before_epoch(t0_plus_150) == 2


def test_trajectory_trajectory_index_after_epoch():
    """Rust: test_trajectory_trajectory_index_after_epoch"""
    # Create a 6-dimensional Trajectory with states at epochs: t0, t0+60s, t0+120s
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        ]
    )

    traj = Trajectory.from_data(epochs, states)

    # Test finding index after t0-30s (should return index 0)
    t0_minus_30 = t0 + (-30.0)
    assert traj.index_after_epoch(t0_minus_30) == 0

    # Test finding index after t0 (should return index 0 - exact match)
    assert traj.index_after_epoch(t0) == 0

    # Test finding index after t0+30s (should return index 1)
    t0_plus_30 = t0 + 30.0
    assert traj.index_after_epoch(t0_plus_30) == 1

    # Test finding index after t0+60s (should return index 1 - exact match)
    assert traj.index_after_epoch(t1) == 1

    # Test finding index after t0+90s (should return index 2)
    t0_plus_90 = t0 + 90.0
    assert traj.index_after_epoch(t0_plus_90) == 2

    # Test finding index after t0+120s (should return index 2 - exact match)
    assert traj.index_after_epoch(t2) == 2

    # Test finding index after t0+150s (should error - after all states)
    t0_plus_150 = t0 + 150.0
    with pytest.raises(Exception):
        traj.index_after_epoch(t0_plus_150)


def test_trajectory_trajectory_state_before_epoch():
    """Rust: test_trajectory_trajectory_state_before_epoch"""
    # Create a Trajectory with distinguishable states at 3 epochs
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        ]
    )

    traj = Trajectory.from_data(epochs, states)

    # Test that state_before_epoch returns correct (epoch, state) tuples
    t0_plus_30 = t0 + 30.0
    epoch, state = traj.state_before_epoch(t0_plus_30)
    assert epoch == t0
    assert state[0] == 1.0

    t0_plus_90 = t0 + 90.0
    epoch, state = traj.state_before_epoch(t0_plus_90)
    assert epoch == t1
    assert state[0] == 11.0

    # Test error case for epoch before all states
    before_t0 = t0 + (-10.0)
    with pytest.raises(Exception):
        traj.state_before_epoch(before_t0)

    # Test that exact matches return the correct state
    epoch, state = traj.state_before_epoch(t1)
    assert epoch == t1
    assert state[0] == 11.0


def test_trajectory_trajectory_state_after_epoch():
    """Rust: test_trajectory_trajectory_state_after_epoch"""
    # Create a Trajectory with distinguishable states at 3 epochs
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        ]
    )

    traj = Trajectory.from_data(epochs, states)

    # Test that state_after_epoch returns correct (epoch, state) tuples
    t0_plus_30 = t0 + 30.0
    epoch, state = traj.state_after_epoch(t0_plus_30)
    assert epoch == t1
    assert state[0] == 11.0

    t0_plus_90 = t0 + 90.0
    epoch, state = traj.state_after_epoch(t0_plus_90)
    assert epoch == t2
    assert state[0] == 21.0

    # Test error case for epoch after all states
    after_t2 = t2 + 10.0
    with pytest.raises(Exception):
        traj.state_after_epoch(after_t2)

    # Verify that exact matches return the correct state
    epoch, state = traj.state_after_epoch(t1)
    assert epoch == t1
    assert state[0] == 11.0


def test_trajectory_set_eviction_policy_max_size():
    """Rust: test_trajectory_set_eviction_policy_max_size"""
    traj = create_test_trajectory()
    assert len(traj) == 3

    traj.set_eviction_policy_max_size(2)
    assert len(traj) == 2
    assert traj.get_eviction_policy() == "KeepCount"


def test_trajectory_set_eviction_policy_max_age():
    """Rust: test_trajectory_set_eviction_policy_max_age"""
    traj = create_test_trajectory()

    # Max age slightly larger than 0.1 days
    traj.set_eviction_policy_max_age(0.11 * 86400.0)
    assert len(traj) == 2
    assert traj.get_eviction_policy() == "KeepWithinDuration"


# Interpolatable Trait Tests


def test_trajectory_interpolatable_get_interpolation_method():
    """Rust: test_trajectory_interpolatable_get_interpolation_method"""
    # Create a trajectory with default Linear interpolation
    traj = Trajectory(6)

    # Test that get_interpolation_method returns Linear
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR

    # Set it to different methods and verify get_interpolation_method returns the correct value
    traj.set_interpolation_method(InterpolationMethod.LINEAR)
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR


def test_trajectory_interpolatable_interpolate_linear():
    """Rust: test_trajectory_interpolatable_interpolate_linear"""
    # Create a 6-dimensional trajectory with 3 states at t0, t0+60s, t0+120s
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [60.0, 120.0, 180.0, 240.0, 300.0, 360.0],
            [120.0, 240.0, 360.0, 480.0, 600.0, 720.0],
        ]
    )

    traj = Trajectory.from_data(epochs, states)

    # Test interpolate_linear at midpoints and exact epochs
    state_at_t0 = traj.interpolate_linear(t0)
    assert state_at_t0[0] == pytest.approx(0.0, abs=1e-10)
    assert state_at_t0[1] == pytest.approx(0.0, abs=1e-10)

    state_at_t1 = traj.interpolate_linear(t1)
    assert state_at_t1[0] == pytest.approx(60.0, abs=1e-10)
    assert state_at_t1[1] == pytest.approx(120.0, abs=1e-10)

    state_at_t2 = traj.interpolate_linear(t2)
    assert state_at_t2[0] == pytest.approx(120.0, abs=1e-10)
    assert state_at_t2[1] == pytest.approx(240.0, abs=1e-10)

    # Test interpolation at midpoint between t0 and t1
    t0_plus_30 = t0 + 30.0
    state_at_midpoint = traj.interpolate_linear(t0_plus_30)
    assert state_at_midpoint[0] == pytest.approx(30.0, abs=1e-10)
    assert state_at_midpoint[1] == pytest.approx(60.0, abs=1e-10)
    assert state_at_midpoint[2] == pytest.approx(90.0, abs=1e-10)
    assert state_at_midpoint[3] == pytest.approx(120.0, abs=1e-10)
    assert state_at_midpoint[4] == pytest.approx(150.0, abs=1e-10)
    assert state_at_midpoint[5] == pytest.approx(180.0, abs=1e-10)

    # Test interpolation at midpoint between t1 and t2
    t1_plus_30 = t1 + 30.0
    state_at_midpoint2 = traj.interpolate_linear(t1_plus_30)
    assert state_at_midpoint2[0] == pytest.approx(90.0, abs=1e-10)
    assert state_at_midpoint2[1] == pytest.approx(180.0, abs=1e-10)
    assert state_at_midpoint2[2] == pytest.approx(270.0, abs=1e-10)
    assert state_at_midpoint2[3] == pytest.approx(360.0, abs=1e-10)
    assert state_at_midpoint2[4] == pytest.approx(450.0, abs=1e-10)
    assert state_at_midpoint2[5] == pytest.approx(540.0, abs=1e-10)

    # Test error case: interpolation outside bounds
    before_t0 = t0 + (-10.0)
    with pytest.raises(Exception):
        traj.interpolate_linear(before_t0)
    after_t2 = t2 + 10.0
    with pytest.raises(Exception):
        traj.interpolate_linear(after_t2)

    # Test edge case: single state trajectory
    single_epoch = [t0]
    single_state = np.array([[100.0, 200.0, 300.0, 400.0, 500.0, 600.0]])
    single_traj = Trajectory.from_data(single_epoch, single_state)

    state_single = single_traj.interpolate_linear(t0)
    assert state_single[0] == pytest.approx(100.0, abs=1e-10)
    assert state_single[1] == pytest.approx(200.0, abs=1e-10)
    assert state_single[2] == pytest.approx(300.0, abs=1e-10)
    assert state_single[3] == pytest.approx(400.0, abs=1e-10)
    assert state_single[4] == pytest.approx(500.0, abs=1e-10)
    assert state_single[5] == pytest.approx(600.0, abs=1e-10)

    # Test error case: interpolation on single state trajectory at different epoch
    different_epoch = t0 + 10.0
    with pytest.raises(Exception):
        single_traj.interpolate_linear(different_epoch)

    # Test error case: interpolation on empty trajectory
    empty_traj = Trajectory(6)
    with pytest.raises(Exception):
        empty_traj.interpolate_linear(t0)


def test_trajectory_interpolatable_interpolate():
    """Rust: test_trajectory_interpolatable_interpolate"""
    # Create a trajectory for testing
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [60.0, 120.0, 180.0, 240.0, 300.0, 360.0],
            [120.0, 240.0, 360.0, 480.0, 600.0, 720.0],
        ]
    )

    traj = Trajectory.from_data(epochs, states)

    # Test that interpolate() with Linear method returns same result as interpolate_linear()
    t0_plus_30 = t0 + 30.0
    state_interpolate = traj.interpolate(t0_plus_30)
    state_interpolate_linear = traj.interpolate_linear(t0_plus_30)

    for i in range(6):
        assert state_interpolate[i] == pytest.approx(
            state_interpolate_linear[i], abs=1e-10
        )


def test_trajectory_interpolate_before_start():
    """Rust test: test_trajectory_interpolate_before_start"""
    # Create a trajectory for testing
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [60.0, 120.0, 180.0, 240.0, 300.0, 360.0],
            [120.0, 240.0, 360.0, 480.0, 600.0, 720.0],
        ]
    )

    traj = Trajectory.from_data(epochs, states)

    # Test interpolation before trajectory start
    before_start = t0 - 10.0
    with pytest.raises(Exception):
        traj.interpolate_linear(before_start)

    # Also test with interpolate() method
    with pytest.raises(Exception):
        traj.interpolate(before_start)


def test_trajectory_interpolate_after_end():
    """Rust test: test_trajectory_interpolate_after_end"""
    # Create a trajectory for testing
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [60.0, 120.0, 180.0, 240.0, 300.0, 360.0],
            [120.0, 240.0, 360.0, 480.0, 600.0, 720.0],
        ]
    )

    traj = Trajectory.from_data(epochs, states)

    # Test interpolation after trajectory end
    after_end = t0 + 130.0
    with pytest.raises(Exception):
        traj.interpolate_linear(after_end)

    # Also test with interpolate() method
    with pytest.raises(Exception):
        traj.interpolate(after_end)


# Covariance Storage Tests


def test_trajectory_enable_covariance_storage():
    """Test enabling covariance storage on a trajectory"""
    traj = Trajectory(6)

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


def test_trajectory_add_with_covariance():
    """Test adding states with covariance matrices"""
    traj = Trajectory(6)

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


def test_trajectory_set_covariance_at():
    """Test setting covariance at a specific index"""
    traj = Trajectory(6)

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


def test_trajectory_covariance_interpolation():
    """Test covariance interpolation at intermediate epochs"""
    traj = Trajectory(6)

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


def test_trajectory_covariance_without_initialization_returns_none():
    """Test that covariance returns None when not initialized"""
    traj = Trajectory(6)

    # Add state without covariance
    t0 = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    traj.add(t0, state)

    # Query covariance should return None
    cov = traj.covariance_at(t0)
    assert cov is None


# Covariance Interpolation Configuration Tests


def test_trajectory_covariance_interpolation_config():
    """Rust test: test_trajectory_covariance_interpolation_config

    Test the CovarianceInterpolationConfig trait implementation.
    """
    # Test default is TwoWasserstein
    traj = Trajectory(6)
    assert (
        traj.get_covariance_interpolation_method()
        == brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )

    # Test with_covariance_interpolation_method builder
    traj = Trajectory(6).with_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )
    assert (
        traj.get_covariance_interpolation_method()
        == brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )

    # Test set_covariance_interpolation_method
    traj = Trajectory(6)
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


def test_trajectory_covariance_interpolation_methods():
    """Rust test: test_trajectory_covariance_interpolation_methods

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
    traj = Trajectory(6)
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


def test_trajectory_covariance_at_exact_epochs():
    """Rust test: test_trajectory_covariance_at_exact_epochs

    Test that covariance_at returns exact values at data points.
    """
    t0 = Epoch.from_jd(2451545.0, brahe.UTC)
    t1 = t0 + 60.0

    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0])

    cov1 = np.eye(6) * 100.0
    cov2 = np.eye(6) * 200.0

    traj = Trajectory(6)
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
