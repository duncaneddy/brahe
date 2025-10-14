"""Tests for DTrajectory in brahe - 1:1 parity with Rust tests"""
import pytest
import numpy as np
import brahe
from brahe import Epoch, DTrajectory, InterpolationMethod


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

    traj = DTrajectory(6)
    for epoch, state in zip(epochs, states):
        traj.add(epoch, state)

    return traj


# Trajectory Trait Tests

def test_dtrajectory_new_with_dimension():
    """Rust: test_dtrajectory_new_with_dimension"""
    # 3
    traj = DTrajectory(3)
    assert traj.dimension() == 3
    assert len(traj) == 0
    assert traj.is_empty()

    # 6
    traj = DTrajectory(6)
    assert traj.dimension() == 6
    assert len(traj) == 0
    assert traj.is_empty()

    # 12
    traj = DTrajectory(12)
    assert traj.dimension() == 12
    assert len(traj) == 0
    assert traj.is_empty()


def test_dtrajectory_new_with_zero_dimension():
    """Rust: test_dtrajectory_new_with_zero_dimension"""
    with pytest.raises(Exception, match="Trajectory dimension must be greater than 0"):
        DTrajectory(0)


def test_dtrajectory_with_interpolation_method():
    """Rust: test_dtrajectory_with_interpolation_method"""
    traj = DTrajectory(12).with_interpolation_method(InterpolationMethod.LINEAR)
    assert traj.dimension() == 12
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR


def test_dtrajectory_with_eviction_policy_max_size_builder():
    """Rust: test_dtrajectory_with_eviction_policy_max_size_builder"""
    # Test builder pattern for max size eviction policy
    traj = DTrajectory(6).with_eviction_policy_max_size(5)

    assert traj.get_eviction_policy() == "KeepCount"
    assert len(traj) == 0


def test_dtrajectory_with_eviction_policy_max_age_builder():
    """Rust: test_dtrajectory_with_eviction_policy_max_age_builder"""
    # Test builder pattern for max age eviction policy
    traj = DTrajectory(6).with_eviction_policy_max_age(300.0)

    assert traj.get_eviction_policy() == "KeepWithinDuration"
    assert len(traj) == 0


def test_dtrajectory_builder_pattern_chaining():
    """Rust: test_dtrajectory_builder_pattern_chaining"""
    # Test chaining multiple builder methods
    traj = DTrajectory(6).with_interpolation_method(InterpolationMethod.LINEAR).with_eviction_policy_max_size(10)

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


def test_dtrajectory_dimension():
    """Rust: test_dtrajectory_dimension"""
    traj = DTrajectory(9)
    assert traj.dimension() == 9

    traj = DTrajectory(4)
    assert traj.dimension() == 4


def test_dtrajectory_interpolatable_set_interpolation_method():
    """Rust: test_dtrajectory_interpolatable_set_interpolation_method"""
    traj = DTrajectory(6)
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR

    traj.set_interpolation_method(InterpolationMethod.LINEAR)
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR


def test_dtrajectory_to_matrix():
    """Rust: test_dtrajectory_to_matrix"""
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


def test_dtrajectory_trajectory_get_eviction_policy():
    """Rust: test_dtrajectory_trajectory_get_eviction_policy"""
    traj = DTrajectory(6)

    # Default is None
    assert traj.get_eviction_policy() == "None"

    # Set to KeepCount
    traj.set_eviction_policy_max_size(10)
    assert traj.get_eviction_policy() == "KeepCount"

    # Set to KeepWithinDuration
    traj.set_eviction_policy_max_age(100.0)
    assert traj.get_eviction_policy() == "KeepWithinDuration"


def test_dtrajectory_apply_eviction_policy_keep_count():
    """Rust: test_dtrajectory_apply_eviction_policy_keep_count"""
    traj = DTrajectory(6).with_eviction_policy_max_size(3)

    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    for i in range(5):
        epoch = t0 + (i * 60.0)
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    # Should only have 3 states due to eviction policy
    assert len(traj) == 3
    assert traj.epoch(0) == t0 + 2.0 * 60.0  # First state should be the third added


def test_dtrajectory_apply_eviction_policy_keep_within_duration():
    """Rust: test_dtrajectory_apply_eviction_policy_keep_within_duration"""
    traj = DTrajectory(6).with_eviction_policy_max_age(86400.0 * 7.0 - 1.0)  # 7 days

    t0 = Epoch.from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    for i in range(10):
        epoch = t0 + (i * 86400.0)  # 1 day apart
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    # Should only have 7 states due to eviction policy
    assert len(traj) == 7
    assert traj.epoch(0) == t0 + 3.0 * 86400.0  # First state should be the fourth added

    # Repeat with an exact 7 days limit
    traj = DTrajectory(6).with_eviction_policy_max_age(86400.0 * 7.0)  # 7 days
    for i in range(10):
        epoch = t0 + (i * 86400.0)
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add(epoch, state)

    # Should still have 8 states due to exact 7 days limit
    assert len(traj) == 8
    assert traj.epoch(0) == t0 + 2.0 * 86400.0  # First state should be the third added


# Default Trait Tests

def test_dtrajectory_default():
    """Rust: test_dtrajectory_default"""
    traj = DTrajectory()
    assert traj.dimension() == 6
    assert len(traj) == 0
    assert traj.is_empty()
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR
    assert traj.get_eviction_policy() == "None"


# Index Trait Tests

def test_dtrajectory_index():
    """Rust: test_dtrajectory_index"""
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


def test_dtrajectory_index_index_out_of_bounds():
    """Rust: test_dtrajectory_index_index_out_of_bounds"""
    traj = create_test_trajectory()
    with pytest.raises(IndexError):
        _ = traj[10]

# IntoIterator Trait Tests

def test_dtrajectory_intoiterator_into_iter():
    """Rust: test_dtrajectory_intoiterator_into_iter"""
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


def test_dtrajectory_intoiterator_into_iter_empty():
    """Rust: test_dtrajectory_intoiterator_into_iter_empty"""
    traj = DTrajectory(6)

    count = 0
    for _ in traj:
        count += 1
    assert count == 0


# Trajectory Trait Tests

def test_dtrajectory_from_data():
    """Rust: test_dtrajectory_from_data"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    # States as 2D array: shape (num_epochs, dimension) = (2, 3)
    states = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    traj = DTrajectory.from_data(epochs, states)
    assert traj.dimension() == 3
    assert len(traj) == 2


def test_dtrajectory_from_data_errors():
    """Rust: test_dtrajectory_from_data_errors"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    # Mismatched: 2 epochs but only 1 state
    states = np.array([
        [1.0, 2.0, 3.0],
    ])

    with pytest.raises(Exception):
        DTrajectory.from_data(epochs, states)

    empty_epochs = []
    empty_states = np.array([]).reshape(0, 3)  # Empty 2D array
    with pytest.raises(Exception):
        DTrajectory.from_data(empty_epochs, empty_states)


def test_dtrajectory_trajectory_add():
    """Rust: test_dtrajectory_trajectory_add"""
    trajectory = DTrajectory(6)

    epoch1 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    trajectory.add(epoch1, state1)
    assert len(trajectory) == 1

    epoch2 = Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, brahe.UTC)
    state2 = np.array([7100e3, 100e3, 50e3, 10.0, 7.6e3, 5.0])

    trajectory.add(epoch2, state2)
    assert len(trajectory) == 2

    np.testing.assert_array_equal(trajectory.state(0), state1)
    np.testing.assert_array_equal(trajectory.state(1), state2)


def test_dtrajectory_trajectory_add_out_of_order():
    """Rust: test_dtrajectory_trajectory_add_out_of_order"""
    trajectory = DTrajectory(6)
    epoch1 = Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([7100e3, 100e3, 60e3, 10.0, 7.6e3, 5.0])

    trajectory.add(epoch1, state1)
    assert len(trajectory) == 1
    assert trajectory.epoch(0) == epoch1
    np.testing.assert_array_equal(trajectory.state(0), state1)

    epoch2 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state2 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    trajectory.add(epoch2, state2)
    assert len(trajectory) == 2
    assert trajectory.epoch(0) == epoch2
    np.testing.assert_array_equal(trajectory.state(0), state2)
    assert trajectory.epoch(1) == epoch1
    np.testing.assert_array_equal(trajectory.state(1), state1)


def test_dtrajectory_trajectory_add_dimension_mismatch():
    """Rust: test_dtrajectory_trajectory_add_dimension_mismatch"""
    trajectory = DTrajectory(6)
    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([7000e3, 0.0, 0.0])  # Dimension 3 instead of 6

    with pytest.raises(Exception):
        trajectory.add(epoch, state)


def test_dtrajectory_trajectory_add_replace():
    """Rust: test_dtrajectory_trajectory_add_replace"""
    trajectory = DTrajectory(6)
    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    trajectory.add(epoch, state1)
    assert len(trajectory) == 1
    np.testing.assert_array_equal(trajectory.state(0), state1)

    state2 = np.array([7100e3, 100e3, 50e3, 10.0, 7.6e3, 5.0])
    trajectory.add(epoch, state2)
    assert len(trajectory) == 1  # Length should remain the same
    np.testing.assert_array_equal(trajectory.state(0), state2)  # State should be replaced


def test_dtrajectory_trajectory_epoch():
    """Rust: test_dtrajectory_trajectory_epoch"""
    traj = create_test_trajectory()

    epoch = traj.epoch(0)
    assert epoch == Epoch.from_jd(2451545.0, brahe.UTC)

    epoch = traj.epoch(1)
    assert epoch == Epoch.from_jd(2451545.1, brahe.UTC)


def test_dtrajectory_trajectory_state():
    """Rust: test_dtrajectory_trajectory_state"""
    traj = create_test_trajectory()

    state = traj.state(0)
    assert state[0] == pytest.approx(7000e3, abs=1.0)

    state = traj.state(1)
    assert state[0] == pytest.approx(7100e3, abs=1.0)


def test_dtrajectory_trajectory_nearest_state():
    """Rust: test_dtrajectory_trajectory_nearest_state"""
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


def test_dtrajectory_trajectory_len():
    """Rust: test_dtrajectory_trajectory_len"""
    traj = create_test_trajectory()
    assert len(traj) == 3

    empty_traj = DTrajectory(6)
    assert len(empty_traj) == 0


def test_dtrajectory_trajectory_is_empty():
    """Rust: test_dtrajectory_trajectory_is_empty"""
    traj = create_test_trajectory()
    assert not traj.is_empty()

    empty_traj = DTrajectory(6)
    assert empty_traj.is_empty()


def test_dtrajectory_trajectory_start_epoch():
    """Rust: test_dtrajectory_trajectory_start_epoch"""
    traj = create_test_trajectory()
    start = traj.start_epoch()
    assert start == Epoch.from_jd(2451545.0, brahe.UTC)

    empty_traj = DTrajectory(6)
    assert empty_traj.start_epoch() is None


def test_dtrajectory_trajectory_end_epoch():
    """Rust: test_dtrajectory_trajectory_end_epoch"""
    traj = create_test_trajectory()
    end = traj.end_epoch()
    assert end == Epoch.from_jd(2451545.2, brahe.UTC)

    empty_traj = DTrajectory(6)
    assert empty_traj.end_epoch() is None


def test_dtrajectory_trajectory_timespan():
    """Rust: test_dtrajectory_trajectory_timespan"""
    traj = create_test_trajectory()
    timespan = traj.timespan()
    assert timespan == pytest.approx(0.2 * 86400.0, abs=1.0)

    empty_traj = DTrajectory(6)
    assert empty_traj.timespan() is None


def test_dtrajectory_trajectory_first():
    """Rust: test_dtrajectory_trajectory_first"""
    traj = create_test_trajectory()
    epoch, state = traj.first()
    assert epoch == Epoch.from_jd(2451545.0, brahe.UTC)
    assert state[0] == pytest.approx(7000e3, abs=1.0)

    empty_traj = DTrajectory(6)
    assert empty_traj.first() is None


def test_dtrajectory_trajectory_last():
    """Rust: test_dtrajectory_trajectory_last"""
    traj = create_test_trajectory()
    epoch, state = traj.last()
    assert epoch == Epoch.from_jd(2451545.2, brahe.UTC)
    assert state[0] == pytest.approx(7200e3, abs=1.0)

    empty_traj = DTrajectory(6)
    assert empty_traj.last() is None


def test_dtrajectory_trajectory_clear():
    """Rust: test_dtrajectory_trajectory_clear"""
    traj = create_test_trajectory()
    assert len(traj) == 3

    traj.clear()
    assert len(traj) == 0
    assert traj.is_empty()


def test_dtrajectory_trajectory_remove_epoch():
    """Rust: test_dtrajectory_trajectory_remove_epoch"""
    traj = create_test_trajectory()
    epoch = Epoch.from_jd(2451545.1, brahe.UTC)

    removed_state = traj.remove_epoch(epoch)
    assert removed_state[0] == pytest.approx(7100e3, abs=1.0)
    assert len(traj) == 2


def test_dtrajectory_trajectory_remove():
    """Rust: test_dtrajectory_trajectory_remove"""
    traj = create_test_trajectory()

    removed_epoch, removed_state = traj.remove(1)
    assert removed_epoch == Epoch.from_jd(2451545.1, brahe.UTC)
    assert removed_state[0] == pytest.approx(7100e3, abs=1.0)
    assert len(traj) == 2


def test_dtrajectory_trajectory_remove_out_of_bounds():
    """Rust: test_dtrajectory_trajectory_remove_out_of_bounds"""
    traj = create_test_trajectory()

    with pytest.raises(Exception):
        traj.remove(10)


def test_dtrajectory_trajectory_get():
    """Rust: test_dtrajectory_trajectory_get"""
    traj = create_test_trajectory()

    epoch, state = traj.get(1)
    assert epoch == Epoch.from_jd(2451545.1, brahe.UTC)
    assert state[0] == pytest.approx(7100e3, abs=1.0)


def test_dtrajectory_trajectory_index_before_epoch():
    """Rust: test_dtrajectory_trajectory_index_before_epoch"""
    # Create a 6-dimensional DTrajectory with states at epochs: t0, t0+60s, t0+120s
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
    ])

    traj = DTrajectory.from_data(epochs, states)

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


def test_dtrajectory_trajectory_index_after_epoch():
    """Rust: test_dtrajectory_trajectory_index_after_epoch"""
    # Create a 6-dimensional DTrajectory with states at epochs: t0, t0+60s, t0+120s
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
    ])

    traj = DTrajectory.from_data(epochs, states)

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


def test_dtrajectory_trajectory_state_before_epoch():
    """Rust: test_dtrajectory_trajectory_state_before_epoch"""
    # Create a DTrajectory with distinguishable states at 3 epochs
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
    ])

    traj = DTrajectory.from_data(epochs, states)

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


def test_dtrajectory_trajectory_state_after_epoch():
    """Rust: test_dtrajectory_trajectory_state_after_epoch"""
    # Create a DTrajectory with distinguishable states at 3 epochs
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
    ])

    traj = DTrajectory.from_data(epochs, states)

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


def test_dtrajectory_set_eviction_policy_max_size():
    """Rust: test_dtrajectory_set_eviction_policy_max_size"""
    traj = create_test_trajectory()
    assert len(traj) == 3

    traj.set_eviction_policy_max_size(2)
    assert len(traj) == 2
    assert traj.get_eviction_policy() == "KeepCount"


def test_dtrajectory_set_eviction_policy_max_age():
    """Rust: test_dtrajectory_set_eviction_policy_max_age"""
    traj = create_test_trajectory()

    # Max age slightly larger than 0.1 days
    traj.set_eviction_policy_max_age(0.11 * 86400.0)
    assert len(traj) == 2
    assert traj.get_eviction_policy() == "KeepWithinDuration"


# Interpolatable Trait Tests

def test_dtrajectory_interpolatable_get_interpolation_method():
    """Rust: test_dtrajectory_interpolatable_get_interpolation_method"""
    # Create a trajectory with default Linear interpolation
    traj = DTrajectory(6)

    # Test that get_interpolation_method returns Linear
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR

    # Set it to different methods and verify get_interpolation_method returns the correct value
    traj.set_interpolation_method(InterpolationMethod.LINEAR)
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR


def test_dtrajectory_interpolatable_interpolate_linear():
    """Rust: test_dtrajectory_interpolatable_interpolate_linear"""
    # Create a 6-dimensional trajectory with 3 states at t0, t0+60s, t0+120s
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [60.0, 120.0, 180.0, 240.0, 300.0, 360.0],
        [120.0, 240.0, 360.0, 480.0, 600.0, 720.0],
    ])

    traj = DTrajectory.from_data(epochs, states)

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
    single_traj = DTrajectory.from_data(single_epoch, single_state)

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
    empty_traj = DTrajectory(6)
    with pytest.raises(Exception):
        empty_traj.interpolate_linear(t0)


def test_dtrajectory_interpolatable_interpolate():
    """Rust: test_dtrajectory_interpolatable_interpolate"""
    # Create a trajectory for testing
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [60.0, 120.0, 180.0, 240.0, 300.0, 360.0],
        [120.0, 240.0, 360.0, 480.0, 600.0, 720.0],
    ])

    traj = DTrajectory.from_data(epochs, states)

    # Test that interpolate() with Linear method returns same result as interpolate_linear()
    t0_plus_30 = t0 + 30.0
    state_interpolate = traj.interpolate(t0_plus_30)
    state_interpolate_linear = traj.interpolate_linear(t0_plus_30)

    for i in range(6):
        assert state_interpolate[i] == pytest.approx(state_interpolate_linear[i], abs=1e-10)
