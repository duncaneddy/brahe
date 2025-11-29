"""
Tests for OrbitTrajectory class in brahe.

These tests provide 1:1 parity with the Rust test suite in src/trajectories/orbit_trajectory.rs
"""

import pytest
import numpy as np
import brahe
from brahe import (
    Epoch,
    TimeSystem,
    OrbitTrajectory,
    OrbitFrame,
    OrbitRepresentation,
    AngleFormat,
    InterpolationMethod,
    R_EARTH,
    DEG2RAD,
    state_koe_to_eci,
    state_eci_to_koe,
    state_eci_to_ecef,
    state_ecef_to_eci,
    state_gcrf_to_itrf,
    state_itrf_to_gcrf,
    state_gcrf_to_eme2000,
    state_eme2000_to_gcrf,
)
from brahe._brahe import PanicException


def create_test_trajectory():
    """Helper function to create a test trajectory (mirrors Rust helper)."""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.DEGREES,
    )

    epoch1 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
    traj.add(epoch1, state1)

    epoch2 = Epoch.from_datetime(2023, 1, 1, 12, 10, 0.0, 0.0, brahe.UTC)
    state2 = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 60.0])
    traj.add(epoch2, state2)

    epoch3 = Epoch.from_datetime(2023, 1, 1, 12, 20, 0.0, 0.0, brahe.UTC)
    state3 = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 75.0])
    traj.add(epoch3, state3)

    return traj


def test_orbittrajectory_new():
    """Rust: test_orbittrajectory_new"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    assert len(traj) == 0
    assert traj.frame == OrbitFrame.ECI
    assert traj.representation == OrbitRepresentation.CARTESIAN
    assert traj.angle_format is None


def test_orbittrajectory_new_invalid_keplerian_none():
    """Rust: test_orbittrajectory_new_invalid_keplerian_none"""
    with pytest.raises(
        ValueError, match="Angle format must be specified for Keplerian"
    ):
        OrbitTrajectory(
            6,
            OrbitFrame.ECI,
            OrbitRepresentation.KEPLERIAN,
        )


def test_orbittrajectory_new_invalid_cartesian_degrees():
    """Rust: test_orbittrajectory_new_invalid_cartesian_degrees"""
    with pytest.raises(ValueError, match="Angle format must be None for Cartesian"):
        OrbitTrajectory(
            6,
            OrbitFrame.ECI,
            OrbitRepresentation.CARTESIAN,
            AngleFormat.DEGREES,
        )


def test_orbittrajectory_new_invalid_cartesian_radians():
    """Rust: test_orbittrajectory_new_invalid_cartesian_radians"""
    with pytest.raises(ValueError, match="Angle format must be None for Cartesian"):
        OrbitTrajectory(
            6,
            OrbitFrame.ECI,
            OrbitRepresentation.CARTESIAN,
            AngleFormat.RADIANS,
        )


def test_orbittrajectory_new_invalid_keplerian_ecef_degrees():
    """Rust: test_orbittrajectory_new_invalid_keplerian_ecef_degrees"""
    with pytest.raises(
        PanicException, match="Keplerian elements should be in ECI frame"
    ):
        OrbitTrajectory(
            6,
            OrbitFrame.ECEF,
            OrbitRepresentation.KEPLERIAN,
            AngleFormat.DEGREES,
        )


def test_orbittrajectory_new_invalid_keplerian_ecef_radians():
    """Rust: test_orbittrajectory_new_invalid_keplerian_ecef_radians"""
    with pytest.raises(
        PanicException, match="Keplerian elements should be in ECI frame"
    ):
        OrbitTrajectory(
            6,
            OrbitFrame.ECEF,
            OrbitRepresentation.KEPLERIAN,
            AngleFormat.RADIANS,
        )


def test_orbittrajectory_new_invalid_keplerian_ecef_none():
    """Rust: test_orbittrajectory_new_invalid_keplerian_ecef_none"""
    with pytest.raises(
        ValueError, match="Angle format must be specified for Keplerian"
    ):
        OrbitTrajectory(
            6,
            OrbitFrame.ECEF,
            OrbitRepresentation.KEPLERIAN,
        )


def test_orbittrajetory_dimension():
    """Rust: test_orbittrajetory_dimension (note the typo!)"""
    traj = create_test_trajectory()
    assert traj.dimension() == 6


def test_orbittrajectory_to_matrix():
    """Rust: test_orbittrajectory_to_matrix"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Convert to matrix
    matrix = traj.to_matrix()

    # Verify dimensions: 3 rows (time points) x 6 columns (state elements)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == 6

    # Verify first row matches first state
    assert matrix[0, 0] == 7000e3
    assert matrix[0, 1] == 0.0
    assert matrix[0, 2] == 0.0
    assert matrix[0, 3] == 0.0
    assert matrix[0, 4] == 7.5e3
    assert matrix[0, 5] == 0.0

    # Verify second row matches second state
    assert matrix[1, 0] == 7100e3
    assert matrix[1, 1] == 1000e3
    assert matrix[1, 2] == 500e3
    assert matrix[1, 3] == 100.0
    assert matrix[1, 4] == 7.6e3
    assert matrix[1, 5] == 50.0

    # Verify third row matches third state
    assert matrix[2, 0] == 7200e3
    assert matrix[2, 1] == 2000e3
    assert matrix[2, 2] == 1000e3
    assert matrix[2, 3] == 200.0
    assert matrix[2, 4] == 7.7e3
    assert matrix[2, 5] == 100.0

    # Verify first column contains first element of each state over time
    assert matrix[0, 0] == 7000e3
    assert matrix[1, 0] == 7100e3
    assert matrix[2, 0] == 7200e3


def test_orbittrajectory_trajectory_add():
    """Rust: test_orbittrajectory_trajectory_add"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Add states in order
    epoch1 = Epoch.from_jd(2451545.0, brahe.UTC)
    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch1, state1)

    epoch3 = Epoch.from_jd(2451545.2, brahe.UTC)
    state3 = np.array([7200e3, 0.0, 0.0, 0.0, 7.7e3, 0.0])
    traj.add(epoch3, state3)

    # Add a state in between
    epoch2 = Epoch.from_jd(2451545.1, brahe.UTC)
    state2 = np.array([7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0])
    traj.add(epoch2, state2)

    assert len(traj) == 3
    assert traj.epoch_at_idx(0).jd() == 2451545.0
    assert traj.epoch_at_idx(1).jd() == 2451545.1
    assert traj.epoch_at_idx(2).jd() == 2451545.2


def test_orbittrajectory_trajectory_state():
    """Rust: test_orbittrajectory_trajectory_state"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Test valid indices
    state0 = traj.state_at_idx(0)
    assert state0[0] == 7000e3

    state1 = traj.state_at_idx(1)
    assert state1[0] == 7100e3

    state2 = traj.state_at_idx(2)
    assert state2[0] == 7200e3

    # Test invalid index
    with pytest.raises(Exception):
        traj.state_at_idx(10)


def test_orbittrajectory_trajectory_epoch():
    """Rust: test_orbittrajectory_trajectory_epoch"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Test valid indices
    epoch0 = traj.epoch_at_idx(0)
    assert epoch0.jd() == 2451545.0

    epoch1 = traj.epoch_at_idx(1)
    assert epoch1.jd() == 2451545.1

    epoch2 = traj.epoch_at_idx(2)
    assert epoch2.jd() == 2451545.2

    # Test invalid index
    with pytest.raises(Exception):
        traj.epoch_at_idx(10)


def test_orbittrajectory_trajectory_nearest_state():
    """Rust: test_orbittrajectory_trajectory_nearest_state"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Test before first epoch
    test_epoch = Epoch.from_jd(2451544.9, brahe.UTC)
    nearest_epoch, nearest_state = traj.nearest_state(test_epoch)
    assert nearest_epoch.jd() == epochs[0].jd()
    assert nearest_state[0] == 7000e3

    # Test after last epoch
    test_epoch = Epoch.from_jd(2451545.3, brahe.UTC)
    nearest_epoch, nearest_state = traj.nearest_state(test_epoch)
    assert nearest_epoch.jd() == epochs[2].jd()
    assert nearest_state[0] == 7200e3

    # Test between epochs
    test_epoch = Epoch.from_jd(2451545.15, brahe.UTC)
    nearest_epoch, nearest_state = traj.nearest_state(test_epoch)
    assert nearest_epoch.jd() == epochs[1].jd()
    assert nearest_state[0] == 7100e3

    # Test exact match
    test_epoch = Epoch.from_jd(2451545.1, brahe.UTC)
    nearest_epoch, nearest_state = traj.nearest_state(test_epoch)
    assert nearest_epoch.jd() == epochs[1].jd()
    assert nearest_state[0] == 7100e3

    # Test just before second epoch
    test_epoch = Epoch.from_jd(2451545.0999, brahe.UTC)
    nearest_epoch, nearest_state = traj.nearest_state(test_epoch)
    assert nearest_epoch.jd() == epochs[1].jd()
    assert nearest_state[0] == 7100e3


def test_orbittrajectory_trajectory_len():
    """Rust: test_orbittrajectory_trajectory_len"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    assert len(traj) == 0
    assert traj.is_empty()

    epoch = Epoch.from_jd(2451545.0, brahe.UTC)
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state)

    assert len(traj) == 1
    assert not traj.is_empty()


def test_orbittrajectory_trajectory_is_empty():
    """Rust: test_orbittrajectory_trajectory_is_empty"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    assert traj.is_empty()

    epoch = Epoch.from_jd(2451545.0, brahe.UTC)
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state)

    assert not traj.is_empty()


def test_orbittrajectory_trajectory_start_epoch():
    """Rust: test_orbittrajectory_trajectory_start_epoch"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    assert traj.start_epoch() is None

    epoch = Epoch.from_jd(2451545.0, brahe.UTC)
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state)

    assert traj.start_epoch().jd() == epoch.jd()


def test_orbittrajectory_trajectory_end_epoch():
    """Rust: test_orbittrajectory_trajectory_end_epoch"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    assert traj.end_epoch() is None

    epoch1 = Epoch.from_jd(2451545.0, brahe.UTC)
    epoch2 = Epoch.from_jd(2451545.1, brahe.UTC)
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch1, state)
    traj.add(epoch2, state)

    assert traj.end_epoch().jd() == epoch2.jd()


def test_orbittrajectory_trajectory_timespan():
    """Rust: test_orbittrajectory_trajectory_timespan"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0],
        ]
    )
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    timespan = traj.timespan()
    assert timespan == pytest.approx(0.1 * 86400.0, abs=1e-5)


def test_orbittrajectory_trajectory_first():
    """Rust: test_orbittrajectory_trajectory_first"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0],
        ]
    )
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    first_epoch, first_state = traj.first()
    assert first_epoch.jd() == epochs[0].jd()
    assert np.array_equal(first_state, np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]))


def test_orbittrajectory_trajectory_last():
    """Rust: test_orbittrajectory_trajectory_last"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0],
        ]
    )
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    last_epoch, last_state = traj.last()
    assert last_epoch.jd() == epochs[1].jd()
    assert np.array_equal(
        last_state, np.array([7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0])
    )


def test_orbittrajectory_trajectory_clear():
    """Rust: test_orbittrajectory_trajectory_clear"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    epoch = Epoch.from_jd(2451545.0, brahe.UTC)
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state)

    assert len(traj) == 1
    traj.clear()
    assert len(traj) == 0


def test_orbittrajectory_trajectory_remove_epoch():
    """Rust: test_orbittrajectory_trajectory_remove_epoch"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0],
        ]
    )
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    removed_state = traj.remove_epoch(epochs[0])
    assert removed_state[0] == 7000e3
    assert len(traj) == 1


def test_orbittrajectory_trajectory_remove():
    """Rust: test_orbittrajectory_trajectory_remove"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0],
        ]
    )
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    removed_epoch, removed_state = traj.remove(0)
    assert removed_epoch.jd() == 2451545.0
    assert removed_state[0] == 7000e3
    assert len(traj) == 1


def test_orbittrajectory_trajectory_get():
    """Rust: test_orbittrajectory_trajectory_get"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0],
        ]
    )
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    epoch, state = traj.get(1)
    assert epoch.jd() == 2451545.1
    assert state[0] == 7100e3


def test_orbittrajectory_trajectory_index_before_epoch():
    """Rust: test_orbittrajectory_trajectory_index_before_epoch"""
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

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

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


def test_orbittrajectory_trajectory_index_after_epoch():
    """Rust: test_orbittrajectory_trajectory_index_after_epoch"""
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

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

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


def test_orbittrajectory_trajectory_state_before_epoch():
    """Rust: test_orbittrajectory_trajectory_state_before_epoch"""
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

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Test that state_before_epoch returns correct (epoch, state) tuples
    t0_plus_30 = t0 + 30.0
    epoch, state = traj.state_before_epoch(t0_plus_30)
    assert epoch.jd() == t0.jd()
    assert state[0] == pytest.approx(1.0, abs=1e-10)

    t0_plus_90 = t0 + 90.0
    epoch, state = traj.state_before_epoch(t0_plus_90)
    assert epoch.jd() == t1.jd()
    assert state[0] == pytest.approx(11.0, abs=1e-10)

    # Test error case for epoch before all states
    before_t0 = t0 + (-10.0)
    with pytest.raises(Exception):
        traj.state_before_epoch(before_t0)

    # Verify it uses the default trait implementation correctly
    epoch, state = traj.state_before_epoch(t2)
    assert epoch.jd() == t2.jd()
    assert state[0] == pytest.approx(21.0, abs=1e-10)


def test_orbittrajectory_trajectory_state_after_epoch():
    """Rust: test_orbittrajectory_trajectory_state_after_epoch"""
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

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Test that state_after_epoch returns correct (epoch, state) tuples
    t0_plus_30 = t0 + 30.0
    epoch, state = traj.state_after_epoch(t0_plus_30)
    assert epoch.jd() == t1.jd()
    assert state[0] == pytest.approx(11.0, abs=1e-10)

    t0_plus_90 = t0 + 90.0
    epoch, state = traj.state_after_epoch(t0_plus_90)
    assert epoch.jd() == t2.jd()
    assert state[0] == pytest.approx(21.0, abs=1e-10)

    # Test error case for epoch after all states
    after_t2 = t2 + 10.0
    with pytest.raises(Exception):
        traj.state_after_epoch(after_t2)

    # Verify it uses the default trait implementation correctly
    epoch, state = traj.state_after_epoch(t0)
    assert epoch.jd() == t0.jd()
    assert state[0] == pytest.approx(1.0, abs=1e-10)


def test_orbittrajectory_trajectory_set_eviction_policy_max_size():
    """Rust: test_orbittrajectory_trajectory_set_eviction_policy_max_size"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

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
    assert first_state[0] == pytest.approx(7000e3 + 2000.0, abs=1.0)

    # Add another state - should still maintain max size
    new_epoch = t0 + 5.0 * 60.0
    new_state = np.array([7000e3 + 5000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(new_epoch, new_state)

    assert len(traj) == 3

    # Test error case
    with pytest.raises(Exception):
        traj.set_eviction_policy_max_size(0)


def test_orbittrajectory_trajectory_set_eviction_policy_max_age():
    """Rust: test_orbittrajectory_trajectory_set_eviction_policy_max_age"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

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

    first_state = traj.state_at_idx(0)
    assert first_state[0] == pytest.approx(7000e3 + 1000.0, abs=1.0)

    # Set max age to 239 seconds
    traj.set_eviction_policy_max_age(239.0)

    assert len(traj) == 4
    first_state = traj.state_at_idx(0)
    assert first_state[0] == pytest.approx(7000e3 + 2000.0, abs=1.0)

    # Test error case
    with pytest.raises(Exception):
        traj.set_eviction_policy_max_age(0.0)
    with pytest.raises(Exception):
        traj.set_eviction_policy_max_age(-10.0)


def test_orbittrajectory_default():
    """Rust: test_orbittrajectory_default"""
    traj = OrbitTrajectory.default()
    assert len(traj) == 0
    assert traj.is_empty()
    assert traj.frame == OrbitFrame.ECI
    assert traj.representation == OrbitRepresentation.CARTESIAN
    assert traj.angle_format is None


def test_orbittrajectory_index_index():
    """Rust: test_orbittrajectory_index_index"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Test indexing returns state vectors
    state0 = traj[0]
    assert state0[0] == 7000e3

    state1 = traj[1]
    assert state1[0] == 7100e3

    state2 = traj[2]
    assert state2[0] == 7200e3


def test_orbittrajectory_index_index_out_of_bounds():
    """Rust: test_orbittrajectory_index_index_out_of_bounds"""
    epochs = [Epoch.from_jd(2451545.0, brahe.UTC)]
    states = np.array([[7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]])
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    with pytest.raises(Exception):
        _ = traj[10]


def test_orbittrajectory_intoiterator_into_iter():
    """Rust: test_orbittrajectory_intoiterator_into_iter"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    count = 0
    for epoch, state in traj:
        if count == 0:
            assert epoch.jd() == 2451545.0
            assert state[0] == 7000e3
        elif count == 1:
            assert epoch.jd() == 2451545.1
            assert state[0] == 7100e3
        elif count == 2:
            assert epoch.jd() == 2451545.2
            assert state[0] == 7200e3
        else:
            pytest.fail("Too many iterations")
        count += 1
    assert count == 3


def test_orbittrajectory_intoiterator_into_iter_empty():
    """Rust: test_orbittrajectory_intoiterator_into_iter_empty"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    count = 0
    for _ in traj:
        count += 1
    assert count == 0


def test_orbittrajectory_iterator_iterator_len():
    """Rust: test_orbittrajectory_iterator_iterator_len"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    iter(traj)
    assert len(traj) == 3


def test_orbittrajectory_interpolatable_set_interpolation_method():
    """Rust: test_orbittrajectory_interpolatable_set_interpolation_method"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR

    traj.set_interpolation_method(InterpolationMethod.LINEAR)
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR


def test_orbittrajectory_interpolatable_get_interpolation_method():
    """Rust: test_orbittrajectory_interpolatable_get_interpolation_method"""
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Test that get_interpolation_method returns Linear
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR

    # Set it to different methods and verify get_interpolation_method returns the correct value
    traj.set_interpolation_method(InterpolationMethod.LINEAR)
    assert traj.get_interpolation_method() == InterpolationMethod.LINEAR


def test_orbittrajectory_interpolatable_interpolate_linear():
    """Rust: test_orbittrajectory_interpolatable_interpolate_linear"""
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

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

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

    # Test interpolation at midpoint between t1 and t2
    t1_plus_30 = t1 + 30.0
    state_at_midpoint2 = traj.interpolate_linear(t1_plus_30)
    assert state_at_midpoint2[0] == pytest.approx(90.0, abs=1e-10)
    assert state_at_midpoint2[1] == pytest.approx(180.0, abs=1e-10)
    assert state_at_midpoint2[2] == pytest.approx(270.0, abs=1e-10)

    # Test edge case: single state trajectory
    single_epoch = [t0]
    single_state = np.array([[100.0, 200.0, 300.0, 400.0, 500.0, 600.0]])
    single_traj = OrbitTrajectory.from_orbital_data(
        single_epoch,
        single_state,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    state_single = single_traj.interpolate_linear(t0)
    assert state_single[0] == pytest.approx(100.0, abs=1e-10)
    assert state_single[1] == pytest.approx(200.0, abs=1e-10)


def test_orbittrajectory_interpolatable_interpolate():
    """Rust: test_orbittrajectory_interpolatable_interpolate"""
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

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Test that interpolate() with Linear method returns same result as interpolate_linear()
    t0_plus_30 = t0 + 30.0
    state_interpolate = traj.interpolate(t0_plus_30)
    state_interpolate_linear = traj.interpolate_linear(t0_plus_30)

    for i in range(6):
        assert state_interpolate[i] == pytest.approx(
            state_interpolate_linear[i], abs=1e-10
        )


def test_orbittrajectory_interpolate_before_start():
    """Rust test: test_orbittrajectory_interpolate_before_start"""
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

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Test interpolation before trajectory start
    before_start = t0 - 10.0
    with pytest.raises(Exception):
        traj.interpolate_linear(before_start)

    # Also test with interpolate() method
    with pytest.raises(Exception):
        traj.interpolate(before_start)


def test_orbittrajectory_interpolate_after_end():
    """Rust test: test_orbittrajectory_interpolate_after_end"""
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

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Test interpolation after trajectory end
    after_end = t0 + 130.0
    with pytest.raises(Exception):
        traj.interpolate_linear(after_end)

    # Also test with interpolate() method
    with pytest.raises(Exception):
        traj.interpolate(after_end)


def test_orbittrajectory_orbitaltrajectory_from_orbital_data():
    """Rust: test_orbittrajectory_orbitaltrajectory_from_orbital_data"""
    epochs = [
        Epoch.from_jd(2451545.0, brahe.UTC),
        Epoch.from_jd(2451545.1, brahe.UTC),
    ]
    states = np.array(
        [
            [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
            [7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0],
        ]
    )

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    assert len(traj) == 2
    assert traj.frame == OrbitFrame.ECI
    assert traj.representation == OrbitRepresentation.CARTESIAN


def test_orbittrajectory_orbitaltrajectory_to_eci():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_eci"""
    tol = 1e-6

    state_base = state_koe_to_eci(
        np.array([R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0]),
        AngleFormat.DEGREES,
    )

    # No transformation needed if already in ECI
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    traj.add(epoch, state_base)

    eci_traj = traj.to_eci()
    assert eci_traj.frame == OrbitFrame.ECI
    assert eci_traj.representation == OrbitRepresentation.CARTESIAN
    assert len(eci_traj) == 1
    epoch_out, state_out = eci_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to ECI - Radians
    kep_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.RADIANS,
    )
    kep_state_rad = state_eci_to_koe(state_base, AngleFormat.RADIANS)
    kep_traj.add(epoch, kep_state_rad)

    eci_from_kep_rad = kep_traj.to_eci()
    assert eci_from_kep_rad.frame == OrbitFrame.ECI
    assert eci_from_kep_rad.representation == OrbitRepresentation.CARTESIAN
    assert len(eci_from_kep_rad) == 1
    epoch_out, state_out = eci_from_kep_rad.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to ECI - Degrees
    kep_traj_deg = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.DEGREES,
    )
    kep_state_deg = state_eci_to_koe(state_base, AngleFormat.DEGREES)
    kep_traj_deg.add(epoch, kep_state_deg)
    eci_from_kep_deg = kep_traj_deg.to_eci()
    assert eci_from_kep_deg.frame == OrbitFrame.ECI
    assert eci_from_kep_deg.representation == OrbitRepresentation.CARTESIAN
    assert len(eci_from_kep_deg) == 1
    epoch_out, state_out = eci_from_kep_deg.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert ECEF to ECI
    ecef_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECEF,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    ecef_state = state_eci_to_ecef(epoch, state_base)
    ecef_traj.add(epoch, ecef_state)
    eci_from_ecef = ecef_traj.to_eci()
    assert eci_from_ecef.frame == OrbitFrame.ECI
    assert eci_from_ecef.representation == OrbitRepresentation.CARTESIAN
    assert len(eci_from_ecef) == 1
    epoch_out, state_out = eci_from_ecef.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)


def test_orbittrajectory_orbitaltrajectory_to_ecef():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_ecef"""
    tol = 1e-6

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state_base = state_eci_to_ecef(
        epoch,
        state_koe_to_eci(
            np.array([R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0]),
            AngleFormat.DEGREES,
        ),
    )

    # No transformation needed if already in ECEF
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECEF,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    traj.add(epoch, state_base)
    ecef_traj = traj.to_ecef()
    assert ecef_traj.frame == OrbitFrame.ECEF
    assert ecef_traj.representation == OrbitRepresentation.CARTESIAN
    assert len(ecef_traj) == 1
    epoch_out, state_out = ecef_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert ECI to ECEF
    eci_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    eci_state = state_ecef_to_eci(epoch, state_base)
    eci_traj.add(epoch, eci_state)
    ecef_from_eci = eci_traj.to_ecef()
    assert ecef_from_eci.frame == OrbitFrame.ECEF
    assert ecef_from_eci.representation == OrbitRepresentation.CARTESIAN
    assert len(ecef_from_eci) == 1
    epoch_out, state_out = ecef_from_eci.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to ECEF - Radians
    kep_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.RADIANS,
    )
    kep_state_rad = state_eci_to_koe(eci_state, AngleFormat.RADIANS)
    kep_traj.add(epoch, kep_state_rad)
    ecef_from_kep_rad = kep_traj.to_ecef()
    assert ecef_from_kep_rad.frame == OrbitFrame.ECEF
    assert ecef_from_kep_rad.representation == OrbitRepresentation.CARTESIAN
    assert len(ecef_from_kep_rad) == 1
    epoch_out, state_out = ecef_from_kep_rad.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to ECEF - Degrees
    kep_traj_deg = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.DEGREES,
    )
    kep_state_deg = state_eci_to_koe(eci_state, AngleFormat.DEGREES)
    kep_traj_deg.add(epoch, kep_state_deg)
    ecef_from_kep_deg = kep_traj_deg.to_ecef()
    assert ecef_from_kep_deg.frame == OrbitFrame.ECEF
    assert ecef_from_kep_deg.representation == OrbitRepresentation.CARTESIAN
    assert len(ecef_from_kep_deg) == 1
    epoch_out, state_out = ecef_from_kep_deg.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)


def test_orbittrajectory_orbitaltrajectory_to_itrf():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_itrf"""
    tol = 1e-6

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state_base = state_gcrf_to_itrf(
        epoch,
        state_koe_to_eci(
            np.array([R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0]),
            AngleFormat.DEGREES,
        ),
    )

    # No transformation needed if already in ITRF
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ITRF,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    traj.add(epoch, state_base)
    itrf_traj = traj.to_itrf()
    assert itrf_traj.frame == OrbitFrame.ITRF
    assert itrf_traj.representation == OrbitRepresentation.CARTESIAN
    assert len(itrf_traj) == 1
    epoch_out, state_out = itrf_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert GCRF to ITRF
    gcrf_traj = OrbitTrajectory(
        6,
        OrbitFrame.GCRF,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    gcrf_state = state_itrf_to_gcrf(epoch, state_base)
    gcrf_traj.add(epoch, gcrf_state)
    itrf_from_gcrf = gcrf_traj.to_itrf()
    assert itrf_from_gcrf.frame == OrbitFrame.ITRF
    assert itrf_from_gcrf.representation == OrbitRepresentation.CARTESIAN
    assert len(itrf_from_gcrf) == 1
    epoch_out, state_out = itrf_from_gcrf.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert EME2000 to ITRF
    eme2000_traj = OrbitTrajectory(
        6,
        OrbitFrame.EME2000,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    eme2000_state = state_gcrf_to_eme2000(gcrf_state)
    eme2000_traj.add(epoch, eme2000_state)
    itrf_from_eme2000 = eme2000_traj.to_itrf()
    assert itrf_from_eme2000.frame == OrbitFrame.ITRF
    assert itrf_from_eme2000.representation == OrbitRepresentation.CARTESIAN
    assert len(itrf_from_eme2000) == 1
    epoch_out, state_out = itrf_from_eme2000.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to ITRF - Radians
    kep_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.RADIANS,
    )
    kep_state_rad = state_eci_to_koe(gcrf_state, AngleFormat.RADIANS)
    kep_traj.add(epoch, kep_state_rad)
    itrf_from_kep_rad = kep_traj.to_itrf()
    assert itrf_from_kep_rad.frame == OrbitFrame.ITRF
    assert itrf_from_kep_rad.representation == OrbitRepresentation.CARTESIAN
    assert len(itrf_from_kep_rad) == 1
    epoch_out, state_out = itrf_from_kep_rad.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to ITRF - Degrees
    kep_traj_deg = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.DEGREES,
    )
    kep_state_deg = state_eci_to_koe(gcrf_state, AngleFormat.DEGREES)
    kep_traj_deg.add(epoch, kep_state_deg)
    itrf_from_kep_deg = kep_traj_deg.to_itrf()
    assert itrf_from_kep_deg.frame == OrbitFrame.ITRF
    assert itrf_from_kep_deg.representation == OrbitRepresentation.CARTESIAN
    assert len(itrf_from_kep_deg) == 1
    epoch_out, state_out = itrf_from_kep_deg.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)


def test_orbittrajectory_orbitaltrajectory_to_gcrf():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_gcrf"""
    tol = 1e-6

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state_base = state_koe_to_eci(
        np.array([R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0]),
        AngleFormat.DEGREES,
    )

    # No transformation needed if already in GCRF
    traj = OrbitTrajectory(
        6,
        OrbitFrame.GCRF,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    traj.add(epoch, state_base)
    gcrf_traj = traj.to_gcrf()
    assert gcrf_traj.frame == OrbitFrame.GCRF
    assert gcrf_traj.representation == OrbitRepresentation.CARTESIAN
    assert len(gcrf_traj) == 1
    epoch_out, state_out = gcrf_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert ITRF to GCRF
    itrf_traj = OrbitTrajectory(
        6,
        OrbitFrame.ITRF,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    itrf_state = state_gcrf_to_itrf(epoch, state_base)
    itrf_traj.add(epoch, itrf_state)
    gcrf_from_itrf = itrf_traj.to_gcrf()
    assert gcrf_from_itrf.frame == OrbitFrame.GCRF
    assert gcrf_from_itrf.representation == OrbitRepresentation.CARTESIAN
    assert len(gcrf_from_itrf) == 1
    epoch_out, state_out = gcrf_from_itrf.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert EME2000 to GCRF
    eme2000_traj = OrbitTrajectory(
        6,
        OrbitFrame.EME2000,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    eme2000_state = state_gcrf_to_eme2000(state_base)
    eme2000_traj.add(epoch, eme2000_state)
    gcrf_from_eme2000 = eme2000_traj.to_gcrf()
    assert gcrf_from_eme2000.frame == OrbitFrame.GCRF
    assert gcrf_from_eme2000.representation == OrbitRepresentation.CARTESIAN
    assert len(gcrf_from_eme2000) == 1
    epoch_out, state_out = gcrf_from_eme2000.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to GCRF - Radians
    kep_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.RADIANS,
    )
    kep_state_rad = state_eci_to_koe(state_base, AngleFormat.RADIANS)
    kep_traj.add(epoch, kep_state_rad)
    gcrf_from_kep_rad = kep_traj.to_gcrf()
    assert gcrf_from_kep_rad.frame == OrbitFrame.GCRF
    assert gcrf_from_kep_rad.representation == OrbitRepresentation.CARTESIAN
    assert len(gcrf_from_kep_rad) == 1
    epoch_out, state_out = gcrf_from_kep_rad.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to GCRF - Degrees
    kep_traj_deg = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.DEGREES,
    )
    kep_state_deg = state_eci_to_koe(state_base, AngleFormat.DEGREES)
    kep_traj_deg.add(epoch, kep_state_deg)
    gcrf_from_kep_deg = kep_traj_deg.to_gcrf()
    assert gcrf_from_kep_deg.frame == OrbitFrame.GCRF
    assert gcrf_from_kep_deg.representation == OrbitRepresentation.CARTESIAN
    assert len(gcrf_from_kep_deg) == 1
    epoch_out, state_out = gcrf_from_kep_deg.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)


def test_orbittrajectory_orbitaltrajectory_to_eme2000():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_eme2000"""
    tol = 1e-6

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state_base = state_gcrf_to_eme2000(
        state_koe_to_eci(
            np.array([R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0]),
            AngleFormat.DEGREES,
        )
    )

    # No transformation needed if already in EME2000
    traj = OrbitTrajectory(
        6,
        OrbitFrame.EME2000,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    traj.add(epoch, state_base)
    eme2000_traj = traj.to_eme2000()
    assert eme2000_traj.frame == OrbitFrame.EME2000
    assert eme2000_traj.representation == OrbitRepresentation.CARTESIAN
    assert len(eme2000_traj) == 1
    epoch_out, state_out = eme2000_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert GCRF to EME2000
    gcrf_traj = OrbitTrajectory(
        6,
        OrbitFrame.GCRF,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    gcrf_state = state_eme2000_to_gcrf(state_base)
    gcrf_traj.add(epoch, gcrf_state)
    eme2000_from_gcrf = gcrf_traj.to_eme2000()
    assert eme2000_from_gcrf.frame == OrbitFrame.EME2000
    assert eme2000_from_gcrf.representation == OrbitRepresentation.CARTESIAN
    assert len(eme2000_from_gcrf) == 1
    epoch_out, state_out = eme2000_from_gcrf.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert ITRF to EME2000
    itrf_traj = OrbitTrajectory(
        6,
        OrbitFrame.ITRF,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    itrf_state = state_gcrf_to_itrf(epoch, gcrf_state)
    itrf_traj.add(epoch, itrf_state)
    eme2000_from_itrf = itrf_traj.to_eme2000()
    assert eme2000_from_itrf.frame == OrbitFrame.EME2000
    assert eme2000_from_itrf.representation == OrbitRepresentation.CARTESIAN
    assert len(eme2000_from_itrf) == 1
    epoch_out, state_out = eme2000_from_itrf.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to EME2000 - Radians
    kep_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.RADIANS,
    )
    kep_state_rad = state_eci_to_koe(gcrf_state, AngleFormat.RADIANS)
    kep_traj.add(epoch, kep_state_rad)
    eme2000_from_kep_rad = kep_traj.to_eme2000()
    assert eme2000_from_kep_rad.frame == OrbitFrame.EME2000
    assert eme2000_from_kep_rad.representation == OrbitRepresentation.CARTESIAN
    assert len(eme2000_from_kep_rad) == 1
    epoch_out, state_out = eme2000_from_kep_rad.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to EME2000 - Degrees
    kep_traj_deg = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.DEGREES,
    )
    kep_state_deg = state_eci_to_koe(gcrf_state, AngleFormat.DEGREES)
    kep_traj_deg.add(epoch, kep_state_deg)
    eme2000_from_kep_deg = kep_traj_deg.to_eme2000()
    assert eme2000_from_kep_deg.frame == OrbitFrame.EME2000
    assert eme2000_from_kep_deg.representation == OrbitRepresentation.CARTESIAN
    assert len(eme2000_from_kep_deg) == 1
    epoch_out, state_out = eme2000_from_kep_deg.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)


def test_orbittrajectory_orbitaltrajectory_to_keplerian_deg():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_keplerian_deg"""
    tol = 1e-6
    DEG2RAD = 0.017453292519943295

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state_kep_deg = np.array([7000e3, 0.01, 97.0, 15.0, 30.0, 45.0])

    # No transformation needed if already in Keplerian Degrees
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.DEGREES,
    )
    traj.add(epoch, state_kep_deg)
    kep_traj = traj.to_keplerian(AngleFormat.DEGREES)
    assert kep_traj.frame == OrbitFrame.ECI
    assert kep_traj.representation == OrbitRepresentation.KEPLERIAN
    assert kep_traj.angle_format == AngleFormat.DEGREES
    assert len(kep_traj) == 1
    epoch_out, state_out = kep_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_deg[i], abs=tol)

    # Convert Keplerian Radians to Keplerian Degrees
    kep_rad_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.RADIANS,
    )
    state_kep_rad = state_kep_deg.copy()
    for i in range(2, 6):
        state_kep_rad[i] = state_kep_deg[i] * DEG2RAD
    kep_rad_traj.add(epoch, state_kep_rad)
    kep_from_rad = kep_rad_traj.to_keplerian(AngleFormat.DEGREES)
    assert kep_from_rad.frame == OrbitFrame.ECI
    assert kep_from_rad.representation == OrbitRepresentation.KEPLERIAN
    assert kep_from_rad.angle_format == AngleFormat.DEGREES
    assert len(kep_from_rad) == 1
    epoch_out, state_out = kep_from_rad.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_deg[i], abs=tol)

    # Convert ECI to Keplerian Degrees
    cart_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    cart_state = state_koe_to_eci(state_kep_deg, AngleFormat.DEGREES)
    cart_traj.add(epoch, cart_state)
    kep_from_cart = cart_traj.to_keplerian(AngleFormat.DEGREES)
    assert kep_from_cart.frame == OrbitFrame.ECI
    assert kep_from_cart.representation == OrbitRepresentation.KEPLERIAN
    assert kep_from_cart.angle_format == AngleFormat.DEGREES
    assert len(kep_from_cart) == 1
    epoch_out, state_out = kep_from_cart.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_deg[i], abs=tol)

    # Convert ECEF to Keplerian Degrees
    ecef_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECEF,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    ecef_state = state_eci_to_ecef(epoch, cart_state)
    ecef_traj.add(epoch, ecef_state)
    kep_from_ecef = ecef_traj.to_keplerian(AngleFormat.DEGREES)
    assert kep_from_ecef.frame == OrbitFrame.ECI
    assert kep_from_ecef.representation == OrbitRepresentation.KEPLERIAN
    assert kep_from_ecef.angle_format == AngleFormat.DEGREES
    assert len(kep_from_ecef) == 1
    epoch_out, state_out = kep_from_ecef.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_deg[i], abs=tol)


def test_orbittrajectory_orbitaltrajectory_to_keplerian_rad():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_keplerian_rad"""
    tol = 1e-6
    DEG2RAD = 0.017453292519943295

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state_kep_deg = np.array([7000e3, 0.01, 97.0, 15.0, 30.0, 45.0])
    state_kep_rad = state_kep_deg.copy()
    for i in range(2, 6):
        state_kep_rad[i] = state_kep_deg[i] * DEG2RAD

    # No transformation needed if already in Keplerian Radians
    traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.RADIANS,
    )
    traj.add(epoch, state_kep_rad)
    kep_traj = traj.to_keplerian(AngleFormat.RADIANS)
    assert kep_traj.frame == OrbitFrame.ECI
    assert kep_traj.representation == OrbitRepresentation.KEPLERIAN
    assert kep_traj.angle_format == AngleFormat.RADIANS
    assert len(kep_traj) == 1
    epoch_out, state_out = kep_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_rad[i], abs=tol)

    # Convert Keplerian Degrees to Keplerian Radians
    kep_deg_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.DEGREES,
    )
    kep_deg_traj.add(epoch, state_kep_deg)
    kep_from_deg = kep_deg_traj.to_keplerian(AngleFormat.RADIANS)
    assert kep_from_deg.frame == OrbitFrame.ECI
    assert kep_from_deg.representation == OrbitRepresentation.KEPLERIAN
    assert kep_from_deg.angle_format == AngleFormat.RADIANS
    assert len(kep_from_deg) == 1
    epoch_out, state_out = kep_from_deg.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_rad[i], abs=tol)

    # Convert ECI to Keplerian Radians
    cart_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    cart_state = state_koe_to_eci(state_kep_deg, AngleFormat.DEGREES)
    cart_traj.add(epoch, cart_state)
    kep_from_cart = cart_traj.to_keplerian(AngleFormat.RADIANS)
    assert kep_from_cart.frame == OrbitFrame.ECI
    assert kep_from_cart.representation == OrbitRepresentation.KEPLERIAN
    assert kep_from_cart.angle_format == AngleFormat.RADIANS
    assert len(kep_from_cart) == 1
    epoch_out, state_out = kep_from_cart.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_rad[i], abs=tol)

    # Convert ECEF to Keplerian Radians
    ecef_traj = OrbitTrajectory(
        6,
        OrbitFrame.ECEF,
        OrbitRepresentation.CARTESIAN,
        None,
    )
    ecef_state = state_eci_to_ecef(epoch, cart_state)
    ecef_traj.add(epoch, ecef_state)
    kep_from_ecef = ecef_traj.to_keplerian(AngleFormat.RADIANS)
    assert kep_from_ecef.frame == OrbitFrame.ECI
    assert kep_from_ecef.representation == OrbitRepresentation.KEPLERIAN
    assert kep_from_ecef.angle_format == AngleFormat.RADIANS
    assert len(kep_from_ecef) == 1
    epoch_out, state_out = kep_from_ecef.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_rad[i], abs=tol)


# StateProvider Tests


def test_orbittrajectory_stateprovider_state_eci_cartesian():
    """Test state() for ECI Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)

    epoch1 = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch1, state1)

    epoch2 = Epoch.from_jd(2451545.5, TimeSystem.UTC)
    state2 = np.array([7200e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0])
    traj.add(epoch2, state2)

    # Query at exact epoch
    state_at_1 = traj.state(epoch1)
    for i in range(6):
        assert state_at_1[i] == pytest.approx(state1[i], abs=1e-6)

    # Query at interpolated epoch
    epoch_mid = Epoch.from_jd(2451545.25, TimeSystem.UTC)
    state_mid = traj.state(epoch_mid)
    # Should be interpolated between state1 and state2
    assert state1[0] < state_mid[0] < state2[0]


def test_orbittrajectory_stateprovider_state_eci():
    """Test state_eci() for ECI Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_eci = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state_eci)

    # Query ECI state
    result = traj.state_eci(epoch)
    for i in range(6):
        assert result[i] == pytest.approx(state_eci[i], abs=1e-6)


def test_orbittrajectory_stateprovider_state_eci_from_keplerian():
    """Test state_eci() for Keplerian trajectory"""
    traj = OrbitTrajectory(
        6, OrbitFrame.ECI, OrbitRepresentation.KEPLERIAN, AngleFormat.DEGREES
    )

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_kep = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
    traj.add(epoch, state_kep)

    # Query ECI Cartesian state
    result = traj.state_eci(epoch)

    # Convert Keplerian to Cartesian manually for comparison
    expected = state_koe_to_eci(state_kep, AngleFormat.DEGREES)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_eci_from_ecef():
    """Test state_eci() for ECEF Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.ECEF, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_ecef = np.array([7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3])
    traj.add(epoch, state_ecef)

    # Query ECI state
    result = traj.state_eci(epoch)

    # Convert ECEF to ECI manually for comparison
    expected = state_ecef_to_eci(epoch, state_ecef)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_ecef():
    """Test state_ecef() for ECEF Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.ECEF, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_ecef = np.array([7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3])
    traj.add(epoch, state_ecef)

    # Query ECEF state
    result = traj.state_ecef(epoch)

    for i in range(6):
        assert result[i] == pytest.approx(state_ecef[i], abs=1e-6)


def test_orbittrajectory_stateprovider_state_ecef_from_eci():
    """Test state_ecef() for ECI Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_eci = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state_eci)

    # Query ECEF state
    result = traj.state_ecef(epoch)

    # Convert ECI to ECEF manually for comparison
    expected = state_eci_to_ecef(epoch, state_eci)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_ecef_from_keplerian():
    """Test state_ecef() for Keplerian trajectory"""
    traj = OrbitTrajectory(
        6, OrbitFrame.ECI, OrbitRepresentation.KEPLERIAN, AngleFormat.DEGREES
    )

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_kep = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
    traj.add(epoch, state_kep)

    # Query ECEF state
    result = traj.state_ecef(epoch)

    # Convert Keplerian -> ECI Cartesian -> ECEF manually for comparison
    state_eci_cart = state_koe_to_eci(state_kep, AngleFormat.DEGREES)
    expected = state_eci_to_ecef(epoch, state_eci_cart)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_gcrf():
    """Test state_gcrf() for GCRF Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.GCRF, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_gcrf = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state_gcrf)

    # Query GCRF state
    result = traj.state_gcrf(epoch)
    for i in range(6):
        assert result[i] == pytest.approx(state_gcrf[i], abs=1e-6)


def test_orbittrajectory_stateprovider_state_gcrf_from_keplerian():
    """Test state_gcrf() for Keplerian trajectory"""
    traj = OrbitTrajectory(
        6, OrbitFrame.ECI, OrbitRepresentation.KEPLERIAN, AngleFormat.DEGREES
    )

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_kep = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
    traj.add(epoch, state_kep)

    # Query GCRF Cartesian state
    result = traj.state_gcrf(epoch)

    # Convert Keplerian to Cartesian manually for comparison
    expected = state_koe_to_eci(state_kep, AngleFormat.DEGREES)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_gcrf_from_itrf():
    """Test state_gcrf() for ITRF Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.ITRF, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_itrf = np.array([7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3])
    traj.add(epoch, state_itrf)

    # Query GCRF state
    result = traj.state_gcrf(epoch)

    # Convert ITRF to GCRF manually for comparison
    expected = state_itrf_to_gcrf(epoch, state_itrf)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_gcrf_from_eme2000():
    """Test state_gcrf() for EME2000 Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.EME2000, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_eme2000 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state_eme2000)

    # Query GCRF state
    result = traj.state_gcrf(epoch)

    # Convert EME2000 to GCRF manually for comparison
    expected = state_eme2000_to_gcrf(state_eme2000)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_itrf():
    """Test state_itrf() for ITRF Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.ITRF, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_itrf = np.array([7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3])
    traj.add(epoch, state_itrf)

    # Query ITRF state
    result = traj.state_itrf(epoch)

    for i in range(6):
        assert result[i] == pytest.approx(state_itrf[i], abs=1e-6)


def test_orbittrajectory_stateprovider_state_itrf_from_keplerian():
    """Test state_itrf() for Keplerian trajectory"""
    traj = OrbitTrajectory(
        6, OrbitFrame.ECI, OrbitRepresentation.KEPLERIAN, AngleFormat.DEGREES
    )

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_kep = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
    traj.add(epoch, state_kep)

    # Query ITRF state
    result = traj.state_itrf(epoch)

    # Convert Keplerian -> GCRF Cartesian -> ITRF manually for comparison
    state_gcrf_cart = state_koe_to_eci(state_kep, AngleFormat.DEGREES)
    expected = state_gcrf_to_itrf(epoch, state_gcrf_cart)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_itrf_from_gcrf():
    """Test state_itrf() for GCRF Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.GCRF, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_gcrf = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state_gcrf)

    # Query ITRF state
    result = traj.state_itrf(epoch)

    # Convert GCRF to ITRF manually for comparison
    expected = state_gcrf_to_itrf(epoch, state_gcrf)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_itrf_from_eme2000():
    """Test state_itrf() for EME2000 Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.EME2000, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_eme2000 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state_eme2000)

    # Query ITRF state
    result = traj.state_itrf(epoch)

    # Convert EME2000 -> GCRF -> ITRF manually for comparison
    state_gcrf = state_eme2000_to_gcrf(state_eme2000)
    expected = state_gcrf_to_itrf(epoch, state_gcrf)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_eme2000():
    """Test state_eme2000() for EME2000 Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.EME2000, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_eme2000 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state_eme2000)

    # Query EME2000 state
    result = traj.state_eme2000(epoch)

    for i in range(6):
        assert result[i] == pytest.approx(state_eme2000[i], abs=1e-6)


def test_orbittrajectory_stateprovider_state_eme2000_from_keplerian():
    """Test state_eme2000() for Keplerian trajectory"""
    traj = OrbitTrajectory(
        6, OrbitFrame.ECI, OrbitRepresentation.KEPLERIAN, AngleFormat.DEGREES
    )

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_kep = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
    traj.add(epoch, state_kep)

    # Query EME2000 state
    result = traj.state_eme2000(epoch)

    # Convert Keplerian -> GCRF Cartesian -> EME2000 manually for comparison
    state_gcrf_cart = state_koe_to_eci(state_kep, AngleFormat.DEGREES)
    expected = state_gcrf_to_eme2000(state_gcrf_cart)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_eme2000_from_gcrf():
    """Test state_eme2000() for GCRF Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.GCRF, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_gcrf = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state_gcrf)

    # Query EME2000 state
    result = traj.state_eme2000(epoch)

    # Convert GCRF to EME2000 manually for comparison
    expected = state_gcrf_to_eme2000(state_gcrf)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_eme2000_from_itrf():
    """Test state_eme2000() for ITRF Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.ITRF, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_itrf = np.array([7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3])
    traj.add(epoch, state_itrf)

    # Query EME2000 state
    result = traj.state_eme2000(epoch)

    # Convert ITRF -> GCRF -> EME2000 manually for comparison
    state_gcrf = state_itrf_to_gcrf(epoch, state_itrf)
    expected = state_gcrf_to_eme2000(state_gcrf)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_stateprovider_state_koe_from_cartesian():
    """Test state_koe() for ECI Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_cart = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state_cart)

    # Query osculating elements in degrees
    result_deg = traj.state_koe(epoch, AngleFormat.DEGREES)

    # Convert Cartesian to Keplerian manually for comparison
    expected_deg = state_eci_to_koe(state_cart, AngleFormat.DEGREES)

    for i in range(6):
        assert result_deg[i] == pytest.approx(expected_deg[i], abs=1e-3)

    # Query osculating elements in radians
    result_rad = traj.state_koe(epoch, AngleFormat.RADIANS)
    expected_rad = state_eci_to_koe(state_cart, AngleFormat.RADIANS)

    for i in range(6):
        assert result_rad[i] == pytest.approx(expected_rad[i], abs=1e-6)


def test_orbittrajectory_stateprovider_state_koe_from_keplerian():
    """Test state_koe() for Keplerian trajectory"""
    traj = OrbitTrajectory(
        6, OrbitFrame.ECI, OrbitRepresentation.KEPLERIAN, AngleFormat.DEGREES
    )

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_kep_deg = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
    traj.add(epoch, state_kep_deg)

    # Query osculating elements in degrees (same as native format)
    result_deg = traj.state_koe(epoch, AngleFormat.DEGREES)

    for i in range(6):
        assert result_deg[i] == pytest.approx(state_kep_deg[i], abs=1e-6)

    # Query osculating elements in radians (requires conversion)
    result_rad = traj.state_koe(epoch, AngleFormat.RADIANS)

    # First two elements unchanged (a, e)
    assert result_rad[0] == pytest.approx(state_kep_deg[0], abs=1e-6)
    assert result_rad[1] == pytest.approx(state_kep_deg[1], abs=1e-9)

    # Angle elements converted
    assert result_rad[2] == pytest.approx(state_kep_deg[2] * DEG2RAD, abs=1e-9)
    assert result_rad[3] == pytest.approx(state_kep_deg[3] * DEG2RAD, abs=1e-9)
    assert result_rad[4] == pytest.approx(state_kep_deg[4] * DEG2RAD, abs=1e-9)
    assert result_rad[5] == pytest.approx(state_kep_deg[5] * DEG2RAD, abs=1e-9)


def test_orbittrajectory_stateprovider_state_koe_from_ecef():
    """Test state_koe() for ECEF Cartesian trajectory"""
    traj = OrbitTrajectory(6, OrbitFrame.ECEF, OrbitRepresentation.CARTESIAN, None)

    epoch = Epoch.from_jd(2451545.0, TimeSystem.UTC)
    state_ecef = np.array([7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3])
    traj.add(epoch, state_ecef)

    # Query osculating elements
    result = traj.state_koe(epoch, AngleFormat.DEGREES)

    # Convert ECEF -> ECI -> Keplerian manually for comparison
    state_eci = state_ecef_to_eci(epoch, state_ecef)
    expected = state_eci_to_koe(state_eci, AngleFormat.DEGREES)

    for i in range(6):
        assert result[i] == pytest.approx(expected[i], abs=1e-3)


def test_orbittrajectory_epochs():
    """Test epochs() returns list of Epoch objects"""
    # Create trajectory with multiple epochs
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Get epochs list
    epochs_list = traj.epochs()

    # Verify it returns a list
    assert isinstance(epochs_list, list)

    # Verify length
    assert len(epochs_list) == 3

    # Verify each element is an Epoch object
    for i, epoch in enumerate(epochs_list):
        assert isinstance(epoch, Epoch)
        assert epoch.jd() == pytest.approx(epochs[i].jd(), abs=1e-10)

    # Test empty trajectory
    empty_traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    empty_epochs = empty_traj.epochs()
    assert isinstance(empty_epochs, list)
    assert len(empty_epochs) == 0


def test_orbittrajectory_epochs_with_frame_transformations():
    """Test epochs() works with epoch objects in frame transformations"""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)

    epoch1 = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch1, state1)

    epoch2 = epoch1 + 60.0
    state2 = np.array([R_EARTH + 500e3, 100e3, 50e3, 10.0, 7600.0, 50.0])
    traj.add(epoch2, state2)

    # Get epochs
    epochs_list = traj.epochs()
    states_list = traj.states()

    # Verify we can use epochs with frame transformation functions
    for i in range(len(epochs_list)):
        epoch = epochs_list[i]
        state_eci = states_list[i, :]

        # This should work without error
        state_ecef = state_eci_to_ecef(epoch, state_eci)
        assert state_ecef is not None
        assert len(state_ecef) == 6


def test_orbittrajectory_states():
    """Test states() returns 2D numpy array"""
    # Create trajectory with multiple states
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.ECI,
        OrbitRepresentation.CARTESIAN,
        None,
    )

    # Get states array
    states_array = traj.states()

    # Verify it returns a numpy array
    assert isinstance(states_array, np.ndarray)

    # Verify shape (N, 6) where N is number of states
    assert states_array.shape == (3, 6)

    # Verify contents match original states
    for i in range(3):
        for j in range(6):
            assert states_array[i, j] == pytest.approx(states[i, j], abs=1e-10)

    # Test empty trajectory - should raise error
    empty_traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    with pytest.raises(RuntimeError, match="Cannot convert empty trajectory to matrix"):
        empty_traj.states()


# ================================
# Identifiable Trait Tests
# ================================


def test_orbittrajectory_identifiable_with_name():
    """Test OrbitTrajectory.with_name() method (mirrors Rust test)."""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj = traj.with_name("Test Trajectory")

    assert traj.get_name() == "Test Trajectory"


def test_orbittrajectory_identifiable_with_id():
    """Test OrbitTrajectory.with_id() method (mirrors Rust test)."""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj = traj.with_id(12345)

    assert traj.get_id() == 12345


def test_orbittrajectory_identifiable_with_uuid():
    """Test OrbitTrajectory.with_new_uuid() method (mirrors Rust test)."""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj = traj.with_new_uuid()

    uuid_str = traj.get_uuid()
    assert uuid_str is not None
    # Verify it's a valid UUID string format
    assert len(uuid_str) == 36
    assert uuid_str.count("-") == 4


def test_orbittrajectory_identifiable_get_methods():
    """Test OrbitTrajectory Identifiable getter methods."""
    # Test that getters return None when not set
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)

    assert traj.get_name() is None
    assert traj.get_id() is None
    assert traj.get_uuid() is None

    # Test after setting values
    traj = traj.with_name("My Trajectory")
    assert traj.get_name() == "My Trajectory"

    traj = traj.with_id(999)
    assert traj.get_id() == 999

    traj = traj.with_new_uuid()
    assert traj.get_uuid() is not None


def test_orbittrajectory_identifiable_builder_chain():
    """Test chaining multiple Identifiable builder methods."""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)

    # Chain multiple builder methods
    traj = traj.with_name("Chained Trajectory").with_id(777).with_new_uuid()

    assert traj.get_name() == "Chained Trajectory"
    assert traj.get_id() == 777
    assert traj.get_uuid() is not None


def test_orbittrajectory_identifiable_preserved_through_conversions():
    """Test that identity is preserved through frame/representation conversions."""
    # Create trajectory with identity
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj = traj.with_name("Conversion Test").with_id(456)

    # Add a state
    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add(epoch, state)

    # Convert to Keplerian
    traj_kep = traj.to_keplerian(AngleFormat.DEGREES)
    assert traj_kep.get_name() == "Conversion Test"
    assert traj_kep.get_id() == 456

    # Convert to ECEF
    traj_ecef = traj.to_ecef()
    assert traj_ecef.get_name() == "Conversion Test"
    assert traj_ecef.get_id() == 456

    # Convert back to ECI
    traj_eci = traj_ecef.to_eci()
    assert traj_eci.get_name() == "Conversion Test"
    assert traj_eci.get_id() == 456


def test_orbittrajectory_identifiable_from_sgp_propagator():
    """Test that identity flows from SGP propagator to trajectory."""
    # Create SGP propagator from 3LE (3-line TLE with name)
    line0 = "ISS (ZARYA)"
    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    propagator = brahe.SGPPropagator.from_3le(line0, line1, line2)

    # Check propagator has the name
    assert propagator.get_name() == "ISS (ZARYA)"

    # Check trajectory inherited the name
    traj = propagator.trajectory
    assert traj.get_name() == "ISS (ZARYA)"
    # NORAD ID should also be propagated (25544)
    assert traj.get_id() == 25544


def test_orbittrajectory_identifiable_name_override():
    """Test overriding name on trajectory after propagator initialization."""
    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    propagator = brahe.SGPPropagator.from_tle(line1, line2)

    # Set name using with_name - this should update both propagator and trajectory
    propagator = propagator.with_name("International Space Station")

    # Check both propagator and trajectory have updated name
    assert propagator.get_name() == "International Space Station"
    assert propagator.trajectory.get_name() == "International Space Station"


def test_sgp_propagator_3le_initialization():
    """Test that name and ID are properly set when initializing from 3LE."""
    # Use a valid 3LE with satellite name in line 0 (ISS)
    line0 = "ISS (ZARYA)"
    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    # Initialize propagator from 3LE
    propagator = brahe.SGPPropagator.from_3le(line0, line1, line2)

    # Verify propagator has the name from line 0
    assert propagator.get_name() == "ISS (ZARYA)"

    # Verify propagator has the NORAD ID from line 1 (25544)
    assert propagator.get_id() == 25544

    # Verify trajectory inherited both name and ID
    traj = propagator.trajectory
    assert traj.get_name() == "ISS (ZARYA)"
    assert traj.get_id() == 25544

    # Verify the name and ID are consistent between propagator and trajectory
    assert propagator.get_name() == traj.get_name()
    assert propagator.get_id() == traj.get_id()


def test_sgp_propagator_tle_no_name():
    """Test that 2-line TLE initialization has ID but no name."""
    # Use a valid 2LE (no name in line 0) - same ISS TLE
    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    # Initialize propagator from 2LE
    propagator = brahe.SGPPropagator.from_tle(line1, line2)

    # Verify propagator has no name (should be None)
    assert propagator.get_name() is None

    # Verify propagator DOES have ID extracted from NORAD catalog number (25544)
    # ID should always be set from TLE, regardless of whether name is provided
    assert propagator.get_id() == 25544

    # Verify trajectory also has no name but DOES have NORAD ID
    traj = propagator.trajectory
    assert traj.get_name() is None
    # Trajectory ID should match propagator ID (25544)
    assert traj.get_id() == 25544

    # Verify consistency
    assert propagator.get_id() == traj.get_id()


def test_from_orbital_data_with_covariances(eop):
    """Test OrbitTrajectory creation with covariances in ECI frame."""
    # Create test data
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    epoch2 = epoch1 + 60.0

    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([brahe.R_EARTH + 500e3, 100.0, 0.0, 0.0, 7.5e3, 0.0])

    cov1 = np.eye(6) * 1000.0
    cov2 = np.eye(6) * 1100.0

    # Create trajectory with covariances
    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch1, epoch2],
        np.array([state1, state2]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov1, cov2]),
    )

    # Verify trajectory was created successfully
    assert len(traj) == 2

    # Verify covariances are retrievable
    retrieved_cov1 = traj.covariance(epoch1)
    assert retrieved_cov1 is not None
    assert np.allclose(retrieved_cov1, cov1, rtol=1e-10)

    retrieved_cov2 = traj.covariance(epoch2)
    assert retrieved_cov2 is not None
    assert np.allclose(retrieved_cov2, cov2, rtol=1e-10)


def test_from_orbital_data_covariances_length_mismatch(eop):
    """Test that mismatched covariance length raises error."""
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    epoch2 = epoch1 + 60.0

    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([brahe.R_EARTH + 500e3, 100.0, 0.0, 0.0, 7.5e3, 0.0])

    cov1 = np.eye(6) * 1000.0
    # Only provide one covariance for two states

    with pytest.raises(Exception) as exc_info:
        brahe.OrbitTrajectory.from_orbital_data(
            [epoch1, epoch2],
            np.array([state1, state2]),
            brahe.OrbitFrame.ECI,
            brahe.OrbitRepresentation.CARTESIAN,
            covariances=np.array([cov1]),  # Length mismatch!
        )

    assert "must match" in str(exc_info.value).lower()


def test_from_orbital_data_covariances_invalid_frame_ecef(eop):
    """Test that covariances with ECEF frame raise error."""
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    cov1 = np.eye(6) * 1000.0

    # Rust panics raise PanicException in PyO3
    with pytest.raises((Exception, brahe.PanicException)) as exc_info:
        brahe.OrbitTrajectory.from_orbital_data(
            [epoch1],
            np.array([state1]),
            brahe.OrbitFrame.ECEF,  # Invalid frame for covariances!
            brahe.OrbitRepresentation.CARTESIAN,
            covariances=np.array([cov1]),
        )

    # Check error message mentions supported frames
    assert (
        "eci" in str(exc_info.value).lower() and "gcrf" in str(exc_info.value).lower()
    )


def test_from_orbital_data_covariances_invalid_frame_itrf(eop):
    """Test that covariances with ITRF frame raise error."""
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    cov1 = np.eye(6) * 1000.0

    # Rust panics raise PanicException in PyO3
    with pytest.raises((Exception, brahe.PanicException)) as exc_info:
        brahe.OrbitTrajectory.from_orbital_data(
            [epoch1],
            np.array([state1]),
            brahe.OrbitFrame.ITRF,  # Invalid frame for covariances!
            brahe.OrbitRepresentation.CARTESIAN,
            covariances=np.array([cov1]),
        )

    # Check error message mentions supported frames
    assert (
        "eci" in str(exc_info.value).lower() and "gcrf" in str(exc_info.value).lower()
    )


def test_add_state_and_covariance(eop):
    """Test adding state with covariance to trajectory."""
    # Create initial trajectory with covariances
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    cov1 = np.eye(6) * 1000.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch1],
        np.array([state1]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov1]),
    )

    # Add new state with covariance
    epoch2 = epoch1 + 60.0
    state2 = np.array([brahe.R_EARTH + 500e3, 100.0, 0.0, 0.0, 7.5e3, 0.0])
    cov2 = np.eye(6) * 1100.0

    traj.add_state_and_covariance(epoch2, state2, cov2)

    # Verify both states and covariances are present
    assert len(traj) == 2

    retrieved_cov1 = traj.covariance(epoch1)
    assert retrieved_cov1 is not None
    assert np.allclose(retrieved_cov1, cov1, rtol=1e-10)

    retrieved_cov2 = traj.covariance(epoch2)
    assert retrieved_cov2 is not None
    assert np.allclose(retrieved_cov2, cov2, rtol=1e-10)


def test_covariance_provider_basic(eop):
    """Test basic covariance retrieval at different epochs."""
    # Create trajectory with covariances
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    epoch2 = epoch1 + 120.0

    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([brahe.R_EARTH + 500e3, 200.0, 0.0, 0.0, 7.5e3, 0.0])

    cov1 = np.eye(6) * 1000.0
    cov2 = np.eye(6) * 2000.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch1, epoch2],
        np.array([state1, state2]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov1, cov2]),
    )

    # Test covariance retrieval at exact epochs
    retrieved_cov1 = traj.covariance(epoch1)
    assert retrieved_cov1 is not None
    assert np.allclose(retrieved_cov1, cov1, rtol=1e-10)

    retrieved_cov2 = traj.covariance(epoch2)
    assert retrieved_cov2 is not None
    assert np.allclose(retrieved_cov2, cov2, rtol=1e-10)

    # Test covariance interpolation at midpoint
    epoch_mid = epoch1 + 60.0
    retrieved_cov_mid = traj.covariance(epoch_mid)
    assert retrieved_cov_mid is not None
    # Verify interpolated covariance is between the two
    assert np.all(retrieved_cov_mid >= cov1 - 1e-6)
    assert np.all(retrieved_cov_mid <= cov2 + 1e-6)

    # Test ECI frame method
    retrieved_cov_eci = traj.covariance_eci(epoch1)
    assert retrieved_cov_eci is not None
    assert np.allclose(retrieved_cov_eci, cov1, rtol=1e-10)

    # Test GCRF frame method (should match ECI for ECI trajectory)
    retrieved_cov_gcrf = traj.covariance_gcrf(epoch1)
    assert retrieved_cov_gcrf is not None


def test_covariance_rtn(eop):
    """Test covariance transformation to RTN frame."""
    # Create trajectory with covariances
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)

    # Define state in ECI
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 1.0e3])

    # Create diagonal covariance
    cov_eci = np.eye(6) * 1000.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov_eci]),
    )

    # Get covariance in RTN frame
    cov_rtn = traj.covariance_rtn(epoch)
    assert cov_rtn is not None

    # Verify it's a valid 6x6 matrix
    assert cov_rtn.shape == (6, 6)

    # Verify it's symmetric (covariance matrices should be symmetric)
    assert np.allclose(cov_rtn, cov_rtn.T, rtol=1e-10)

    # Verify diagonal elements are positive (variances must be positive)
    assert np.all(np.diag(cov_rtn) > 0)


def test_covariance_error_for_trajectory_without_covariances(eop):
    """Test that covariance methods raise an error for trajectories without covariances."""
    # Create trajectory WITHOUT covariances
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        # No covariances parameter
    )

    # All covariance methods should raise an error when covariances not initialized
    with pytest.raises(Exception, match="covariance tracking was not enabled"):
        traj.covariance(epoch)
    with pytest.raises(Exception, match="covariance tracking was not enabled"):
        traj.covariance_eci(epoch)
    with pytest.raises(Exception, match="covariance tracking was not enabled"):
        traj.covariance_gcrf(epoch)
    with pytest.raises(Exception, match="covariance tracking was not enabled"):
        traj.covariance_rtn(epoch)


def test_covariance_interpolation_method_two_wasserstein(eop):
    """Test TwoWasserstein covariance interpolation method."""
    # Create trajectory with covariances
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    epoch2 = epoch1 + 120.0

    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([brahe.R_EARTH + 500e3, 200.0, 0.0, 0.0, 7.5e3, 0.0])

    cov1 = np.eye(6) * 1000.0
    cov2 = np.eye(6) * 2000.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch1, epoch2],
        np.array([state1, state2]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov1, cov2]),
    )

    # TwoWasserstein is the default and currently uses linear interpolation as a stub
    traj.set_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )

    # Verify method was set
    method = traj.get_covariance_interpolation_method()
    assert method == brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN

    # Test interpolation at midpoint
    epoch_mid = epoch1 + 60.0
    retrieved_cov_mid = traj.covariance(epoch_mid)
    assert retrieved_cov_mid is not None

    # Verify interpolated value is between endpoints
    assert np.all(retrieved_cov_mid >= np.minimum(cov1, cov2))
    assert np.all(retrieved_cov_mid <= np.maximum(cov1, cov2))


def test_covariance_interpolation_method_matrix_square_root(eop):
    """Test matrix square root covariance interpolation method."""
    # Create trajectory with covariances
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    epoch2 = epoch1 + 120.0

    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([brahe.R_EARTH + 500e3, 200.0, 0.0, 0.0, 7.5e3, 0.0])

    cov1 = np.eye(6) * 1000.0
    cov2 = np.eye(6) * 2000.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch1, epoch2],
        np.array([state1, state2]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov1, cov2]),
    )

    # Set matrix square root interpolation method
    traj.set_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )

    # Verify method was set
    method = traj.get_covariance_interpolation_method()
    assert method == brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT

    # Test interpolation at midpoint
    epoch_mid = epoch1 + 60.0
    retrieved_cov_mid = traj.covariance(epoch_mid)
    assert retrieved_cov_mid is not None

    # Verify interpolated value is between endpoints
    assert np.all(retrieved_cov_mid >= np.minimum(cov1, cov2))
    assert np.all(retrieved_cov_mid <= np.maximum(cov1, cov2))


def test_with_covariance_interpolation_method_builder(eop):
    """Test builder pattern for setting covariance interpolation method."""
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    cov = np.eye(6) * 1000.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov]),
    ).with_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )

    # Verify method was set via builder pattern
    method = traj.get_covariance_interpolation_method()
    assert method == brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT


def test_covariance_eci(eop):
    """Test covariance_eci method returns covariance in ECI frame."""
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    cov = np.eye(6) * 100.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov]),
    )

    result = traj.covariance_eci(epoch)
    assert result is not None

    # Verify diagonal elements
    assert result[0, 0] == pytest.approx(100.0, rel=1e-6)
    assert result[1, 1] == pytest.approx(100.0, rel=1e-6)
    assert result[2, 2] == pytest.approx(100.0, rel=1e-6)


def test_covariance_gcrf(eop):
    """Test covariance_gcrf method returns covariance in GCRF frame."""
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    cov = np.eye(6) * 100.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.GCRF,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov]),
    )

    result = traj.covariance_gcrf(epoch)
    assert result is not None

    # Verify diagonal elements
    assert result[0, 0] == pytest.approx(100.0, rel=1e-6)
    assert result[1, 1] == pytest.approx(100.0, rel=1e-6)
    assert result[2, 2] == pytest.approx(100.0, rel=1e-6)


def test_covariance_eci_from_eme2000_frame(eop):
    """Test covariance_eci with trajectory in EME2000 frame."""
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    # Create diagonal covariance matrix in EME2000 frame
    cov_eme2000 = np.eye(6) * 100.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.EME2000,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov_eme2000]),
    )

    # Get covariance in ECI frame (should be transformed)
    result = traj.covariance_eci(epoch)
    assert result is not None

    cov_eci = result

    # Verify covariance is symmetric (should be preserved by transformation)
    for i in range(6):
        for j in range(6):
            assert cov_eci[i, j] == pytest.approx(cov_eci[j, i], abs=1e-10)

    # Verify diagonal elements are positive (positive-definiteness check)
    for i in range(6):
        assert cov_eci[i, i] > 0.0

    # Verify diagonal elements are preserved (EME2000-GCRF bias is very small)
    for i in range(6):
        assert cov_eci[i, i] == pytest.approx(100.0, abs=1e-3)


def test_covariance_gcrf_from_eme2000_frame(eop):
    """Test covariance_gcrf with trajectory in EME2000 frame."""
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    cov_eme2000 = np.eye(6) * 100.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.EME2000,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov_eme2000]),
    )

    # covariance_gcrf should delegate to covariance_eci
    result_gcrf = traj.covariance_gcrf(epoch)
    result_eci = traj.covariance_eci(epoch)

    assert result_gcrf is not None
    assert result_eci is not None

    # GCRF and ECI should be identical for EME2000 transformation
    for i in range(6):
        for j in range(6):
            assert result_gcrf[i, j] == pytest.approx(result_eci[i, j], abs=1e-12)


def test_covariance_rtn_from_eme2000_frame(eop):
    """Test covariance_rtn with trajectory in EME2000 frame."""
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    cov_eme2000 = np.eye(6) * 100.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.EME2000,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov_eme2000]),
    )

    # Get covariance in RTN frame (should go EME2000 -> ECI -> RTN)
    result = traj.covariance_rtn(epoch)
    assert result is not None

    cov_rtn = result

    # Verify covariance is symmetric
    for i in range(6):
        for j in range(6):
            assert cov_rtn[i, j] == pytest.approx(cov_rtn[j, i], abs=1e-10)

    # Verify diagonal elements are positive
    for i in range(6):
        assert cov_rtn[i, i] > 0.0

    # RTN covariance should be non-trivial (not identity)
    # Either diagonal differs from 100 or off-diagonal is non-zero
    is_non_identity = any(abs(cov_rtn[i, i] - 100.0) > 1e-6 for i in range(6)) or any(
        abs(cov_rtn[i, j]) > 1e-6 for i in range(6) for j in range(6) if i != j
    )
    assert is_non_identity, "RTN transformation should produce non-identity matrix"


def test_covariance_interpolatable_trait_methods(eop):
    """Test trait methods: getter, setter, and builder pattern."""
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    cov = np.eye(6) * 100.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov]),
    )

    # Test getter - default should be TwoWasserstein
    method = traj.get_covariance_interpolation_method()
    assert method == brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN

    # Test setter
    traj.set_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )
    method = traj.get_covariance_interpolation_method()
    assert method == brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT

    # Test builder pattern
    traj2 = traj.with_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )
    method = traj2.get_covariance_interpolation_method()
    assert method == brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN


def test_covariance_interpolation_edge_cases_matrix_square_root(eop):
    """Test covariance interpolation edge cases with MatrixSquareRoot method."""
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    epoch2 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 10, 0.0, 0.0, brahe.UTC)
    epoch3 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 20, 0.0, 0.0, brahe.UTC)

    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state3 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    cov1 = np.eye(6) * 100.0
    cov2 = np.eye(6) * 200.0
    cov3 = np.eye(6) * 300.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch1, epoch2, epoch3],
        np.array([state1, state2, state3]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov1, cov2, cov3]),
    ).with_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )

    # Test at exact epoch
    result_exact = traj.covariance(epoch2)
    assert result_exact is not None
    assert result_exact[0, 0] == pytest.approx(200.0, rel=1e-6)

    # Test halfway between epoch1 and epoch2 (should be interpolated)
    epoch_halfway = brahe.Epoch.from_datetime(2024, 1, 1, 0, 5, 0.0, 0.0, brahe.UTC)
    result_halfway = traj.covariance(epoch_halfway)
    assert result_halfway is not None
    # Verify interpolation gives value between endpoints
    assert 100.0 < result_halfway[0, 0] < 200.0

    # Test before data range (should raise an error)
    epoch_before = brahe.Epoch.from_datetime(2023, 12, 31, 23, 50, 0.0, 0.0, brahe.UTC)
    with pytest.raises(Exception, match="before.*trajectory start"):
        traj.covariance(epoch_before)

    # Test after data range (should raise an error)
    epoch_after = brahe.Epoch.from_datetime(2024, 1, 1, 0, 30, 0.0, 0.0, brahe.UTC)
    with pytest.raises(
        Exception, match="(outside covariance data range|after trajectory end)"
    ):
        traj.covariance(epoch_after)


def test_covariance_interpolation_edge_cases_two_wasserstein(eop):
    """Test covariance interpolation edge cases with TwoWasserstein method."""
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    epoch2 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 10, 0.0, 0.0, brahe.UTC)
    epoch3 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 20, 0.0, 0.0, brahe.UTC)

    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state3 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    cov1 = np.eye(6) * 100.0
    cov2 = np.eye(6) * 200.0
    cov3 = np.eye(6) * 300.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch1, epoch2, epoch3],
        np.array([state1, state2, state3]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov1, cov2, cov3]),
    ).with_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )

    # Test at exact epoch
    result_exact = traj.covariance(epoch2)
    assert result_exact is not None
    assert result_exact[0, 0] == pytest.approx(200.0, rel=1e-6)

    # Test halfway between epoch1 and epoch2 (should be interpolated)
    epoch_halfway = brahe.Epoch.from_datetime(2024, 1, 1, 0, 5, 0.0, 0.0, brahe.UTC)
    result_halfway = traj.covariance(epoch_halfway)
    assert result_halfway is not None
    # Verify interpolation gives value between endpoints
    assert 100.0 < result_halfway[0, 0] < 200.0

    # Test before data range (should raise an error)
    epoch_before = brahe.Epoch.from_datetime(2023, 12, 31, 23, 50, 0.0, 0.0, brahe.UTC)
    with pytest.raises(Exception, match="before.*trajectory start"):
        traj.covariance(epoch_before)

    # Test after data range (should raise an error)
    epoch_after = brahe.Epoch.from_datetime(2024, 1, 1, 0, 30, 0.0, 0.0, brahe.UTC)
    with pytest.raises(
        Exception, match="(outside covariance data range|after trajectory end)"
    ):
        traj.covariance(epoch_after)


def test_covariance_interpolation_methods_comparison(eop):
    """Test both interpolation methods produce valid symmetric PSD matrices."""
    epoch1 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    epoch2 = brahe.Epoch.from_datetime(2024, 1, 1, 0, 10, 0.0, 0.0, brahe.UTC)

    state1 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    state2 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

    cov1 = np.eye(6) * 100.0
    cov2 = np.eye(6) * 200.0

    # Test both interpolation methods
    traj_wasserstein = brahe.OrbitTrajectory.from_orbital_data(
        [epoch1, epoch2],
        np.array([state1, state2]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov1, cov2]),
    ).with_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )

    traj_matrix_sqrt = brahe.OrbitTrajectory.from_orbital_data(
        [epoch1, epoch2],
        np.array([state1, state2]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov1, cov2]),
    ).with_covariance_interpolation_method(
        brahe.CovarianceInterpolationMethod.MATRIX_SQUARE_ROOT
    )

    # Query at midpoint
    epoch_mid = brahe.Epoch.from_datetime(2024, 1, 1, 0, 5, 0.0, 0.0, brahe.UTC)

    cov_wasserstein = traj_wasserstein.covariance(epoch_mid)
    cov_matrix_sqrt = traj_matrix_sqrt.covariance(epoch_mid)

    assert cov_wasserstein is not None
    assert cov_matrix_sqrt is not None

    # Both should be symmetric and positive-definite
    for i in range(6):
        assert cov_wasserstein[i, i] > 0.0
        assert cov_matrix_sqrt[i, i] > 0.0
        for j in range(6):
            assert cov_wasserstein[i, j] == pytest.approx(
                cov_wasserstein[j, i], abs=1e-10
            )
            assert cov_matrix_sqrt[i, j] == pytest.approx(
                cov_matrix_sqrt[j, i], abs=1e-10
            )

    # Both methods should give values in reasonable range
    assert 100.0 < cov_wasserstein[0, 0] < 200.0
    assert 100.0 < cov_matrix_sqrt[0, 0] < 200.0

    # For diagonal matrices, both methods should give identical results
    assert cov_wasserstein[0, 0] == pytest.approx(cov_matrix_sqrt[0, 0], rel=1e-6)


def test_covariance_single_point_trajectory(eop):
    """Test covariance with single data point trajectory."""
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    state = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    cov = np.eye(6) * 100.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov]),
    )

    # Exact epoch should return covariance
    result_exact = traj.covariance(epoch)
    assert result_exact is not None
    assert result_exact[0, 0] == pytest.approx(100.0, rel=1e-6)

    # Different epoch should raise an error (no interpolation possible with single point)
    epoch_later = epoch + 60.0
    with pytest.raises(
        Exception, match="(outside covariance data range|after trajectory end)"
    ):
        traj.covariance(epoch_later)


def test_covariance_rtn_elliptical_orbit(eop):
    """Test RTN covariance for elliptical inclined orbit."""
    epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)

    # Create elliptical inclined orbit state
    # a = R_EARTH + 600km, e = 0.2, i = 63.4 deg
    a = brahe.R_EARTH + 600e3
    e = 0.2
    i = np.radians(63.4)
    raan = np.radians(45.0)
    argp = np.radians(30.0)
    nu = 0.0  # True anomaly

    oe = np.array([a, e, i, raan, argp, nu])
    state = brahe.state_koe_to_eci(oe, brahe.AngleFormat.RADIANS)

    cov = np.eye(6) * 100.0

    traj = brahe.OrbitTrajectory.from_orbital_data(
        [epoch],
        np.array([state]),
        brahe.OrbitFrame.ECI,
        brahe.OrbitRepresentation.CARTESIAN,
        covariances=np.array([cov]),
    )

    # Get covariance in RTN frame
    result = traj.covariance_rtn(epoch)
    assert result is not None

    cov_rtn = result

    # Verify RTN covariance is symmetric
    for i in range(6):
        for j in range(6):
            assert cov_rtn[i, j] == pytest.approx(cov_rtn[j, i], abs=1e-10)

    # Verify diagonal elements are positive
    for i in range(6):
        assert cov_rtn[i, i] > 0.0

    # RTN transformation should produce different values than identity
    differs_from_identity = any(
        abs(cov_rtn[i, i] - 100.0) > 1e-6 for i in range(6)
    ) or any(abs(cov_rtn[i, j]) > 1e-6 for i in range(6) for j in range(6) if i != j)
    assert differs_from_identity


# ========================
# Interpolation Method Tests
# ========================


def test_orbittrajectory_interpolation_linear():
    """Test linear interpolation on OrbitTrajectory."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj.set_interpolation_method(InterpolationMethod.LINEAR)
    traj.add(t0, np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]))
    traj.add(t1, np.array([7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]))

    # Interpolate at midpoint
    t_mid = t0 + 30.0
    result = traj.interpolate(t_mid)

    # Linear interpolation should give exact midpoint
    assert result[0] == pytest.approx(7030e3, abs=1.0)


def test_orbittrajectory_interpolation_lagrange_degree2():
    """Test Lagrange degree 2 interpolation on OrbitTrajectory."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj.set_interpolation_method(InterpolationMethod.lagrange(2))

    # Add 3 points with quadratic position profile: x = 7000e3 + 1000*t + 0.5*t^2
    for i in range(3):
        dt = i * 30.0
        x = 7000e3 + 1000.0 * dt + 0.5 * dt * dt
        traj.add(t0 + dt, np.array([x, 0.0, 0.0, 0.0, 7.5e3, 0.0]))

    # Interpolate at t = 45s
    t_query = t0 + 45.0
    result = traj.interpolate(t_query)

    # Expected: 7000e3 + 1000*45 + 0.5*45^2 = 7046012.5
    expected_x = 7000e3 + 1000.0 * 45.0 + 0.5 * 45.0 * 45.0
    assert result[0] == pytest.approx(expected_x, abs=1.0)


def test_orbittrajectory_interpolation_lagrange_degree3():
    """Test Lagrange degree 3 interpolation on OrbitTrajectory."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj.set_interpolation_method(InterpolationMethod.lagrange(3))

    # Add 4 points with cubic profile
    for i in range(4):
        dt = i * 30.0
        x = 7000e3 + 100.0 * dt + 0.1 * dt * dt + 0.001 * dt * dt * dt
        traj.add(t0 + dt, np.array([x, 0.0, 0.0, 0.0, 7.5e3, 0.0]))

    # Interpolate at t = 45s
    t_query = t0 + 45.0
    result = traj.interpolate(t_query)

    expected_x = 7000e3 + 100.0 * 45.0 + 0.1 * 45.0 * 45.0 + 0.001 * 45.0**3
    assert result[0] == pytest.approx(expected_x, abs=1.0)


def test_orbittrajectory_interpolation_hermite_cubic():
    """Test Hermite cubic interpolation on OrbitTrajectory."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj.set_interpolation_method(InterpolationMethod.HERMITE_CUBIC)

    # State 0: position = (7000e3, 0, 0), velocity = (100, 7500, 0)
    traj.add(t0, np.array([7000e3, 0.0, 0.0, 100.0, 7500.0, 0.0]))
    # State 1: position = (7006e3, 450e3, 0), velocity = (100, 7500, 0)
    traj.add(t1, np.array([7006e3, 450e3, 0.0, 100.0, 7500.0, 0.0]))

    # Interpolate at midpoint
    t_mid = t0 + 30.0
    result = traj.interpolate(t_mid)

    # Hermite cubic should give smooth interpolation
    assert result[0] == pytest.approx(7003e3, abs=100.0)
    assert result[1] == pytest.approx(225e3, abs=100.0)


def test_orbittrajectory_interpolation_lagrange_vs_linear_different():
    """Test that Lagrange interpolation gives different (better) results than linear for nonlinear data."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)

    # Create two trajectories with the same quadratic data
    traj_linear = OrbitTrajectory(
        6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None
    )
    traj_lagrange = OrbitTrajectory(
        6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None
    )

    traj_linear.set_interpolation_method(InterpolationMethod.LINEAR)
    traj_lagrange.set_interpolation_method(InterpolationMethod.lagrange(2))

    # Add 3 points with quadratic profile
    for i in range(3):
        dt = i * 60.0
        x = 7000e3 + dt * dt  # Quadratic, not linear
        state = np.array([x, 0.0, 0.0, 0.0, 7500.0, 0.0])
        traj_linear.add(t0 + dt, state)
        traj_lagrange.add(t0 + dt, state)

    # Interpolate at t = 90s
    t_query = t0 + 90.0
    result_linear = traj_linear.interpolate(t_query)
    result_lagrange = traj_lagrange.interpolate(t_query)

    expected_exact = 7000e3 + 90.0 * 90.0

    # Lagrange should be closer to exact value
    linear_error = abs(result_linear[0] - expected_exact)
    lagrange_error = abs(result_lagrange[0] - expected_exact)

    assert lagrange_error < linear_error, (
        f"Lagrange error ({lagrange_error}) should be less than linear error ({linear_error})"
    )


# ========================
# Acceleration Storage Tests
# ========================


def test_orbittrajectory_acceleration_storage_disabled_by_default():
    """Test that acceleration storage is disabled by default."""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    assert not traj.has_accelerations()


def test_orbittrajectory_enable_acceleration_storage():
    """Test enabling acceleration storage."""
    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj.enable_acceleration_storage(3)
    assert traj.has_accelerations()


def test_orbittrajectory_add_with_acceleration():
    """Test adding states with accelerations."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj.enable_acceleration_storage(3)

    state = np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    acc = np.array([-9.0, 0.0, 0.0])
    traj.add_with_acceleration(t0, state, acc)

    assert len(traj) == 1

    retrieved_acc = traj.acceleration_at_idx(0)
    assert retrieved_acc[0] == pytest.approx(-9.0, abs=1e-10)
    assert retrieved_acc[1] == pytest.approx(0.0, abs=1e-10)
    assert retrieved_acc[2] == pytest.approx(0.0, abs=1e-10)


def test_orbittrajectory_set_acceleration_at():
    """Test setting acceleration at a specific index."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj.enable_acceleration_storage(3)

    state = np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    traj.add(t0, state)

    # Initially acceleration should be zero
    acc_before = traj.acceleration_at_idx(0)
    assert acc_before[0] == pytest.approx(0.0, abs=1e-10)

    # Set acceleration
    new_acc = np.array([-8.5, 0.1, -0.05])
    traj.set_acceleration_at(0, new_acc)

    acc_after = traj.acceleration_at_idx(0)
    assert acc_after[0] == pytest.approx(-8.5, abs=1e-10)
    assert acc_after[1] == pytest.approx(0.1, abs=1e-10)
    assert acc_after[2] == pytest.approx(-0.05, abs=1e-10)


def test_orbittrajectory_acceleration_at_idx_no_storage():
    """Test that acceleration_at_idx returns None when storage is not enabled."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    # Acceleration storage NOT enabled
    traj.add(t0, np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0]))

    result = traj.acceleration_at_idx(0)
    assert result is None


def test_orbittrajectory_hermite_quintic_with_accelerations():
    """Test HermiteQuintic interpolation with stored accelerations."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    t1 = t0 + 60.0

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    traj.enable_acceleration_storage(3)
    traj.set_interpolation_method(InterpolationMethod.HERMITE_QUINTIC)

    # Add states with accelerations
    state0 = np.array([7000e3, 0.0, 0.0, 100.0, 7500.0, 0.0])
    acc0 = np.array([1.0, 0.0, 0.0])
    traj.add_with_acceleration(t0, state0, acc0)

    state1 = np.array([7006e3 + 1800.0, 450e3, 0.0, 160.0, 7500.0, 0.0])
    acc1 = np.array([1.0, 0.0, 0.0])
    traj.add_with_acceleration(t1, state1, acc1)

    # Interpolate at midpoint
    t_mid = t0 + 30.0
    result = traj.interpolate(t_mid)

    # Quintic Hermite should give smooth result
    assert result[0] > 7000e3 and result[0] < 7008e3


def test_orbittrajectory_hermite_quintic_finite_difference():
    """Test HermiteQuintic interpolation with finite difference fallback."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    # No acceleration storage - will use finite differences
    traj.set_interpolation_method(InterpolationMethod.HERMITE_QUINTIC)

    # Add 3 points for finite difference approximation
    # Use parabolic motion in x: x = 7000e3 + 100*t + 0.5*t^2
    for i in range(3):
        dt = i * 30.0
        x = 7000e3 + 100.0 * dt + 0.5 * dt * dt
        vx = 100.0 + dt  # velocity = derivative = 100 + t
        traj.add(t0 + dt, np.array([x, 0.0, 0.0, vx, 7500.0, 0.0]))

    # Interpolate at t = 45s
    t_query = t0 + 45.0
    result = traj.interpolate(t_query)

    # Expected position: 7000e3 + 100*45 + 0.5*45^2 = 7005512.5
    expected_x = 7000e3 + 100.0 * 45.0 + 0.5 * 45.0 * 45.0
    assert result[0] == pytest.approx(expected_x, abs=100.0)


def test_orbittrajectory_add_with_acceleration_requires_enabled():
    """Test that add_with_acceleration raises error if storage not enabled."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    # Storage NOT enabled

    state = np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    acc = np.array([-9.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Acceleration storage is not enabled"):
        traj.add_with_acceleration(t0, state, acc)


def test_orbittrajectory_set_acceleration_requires_enabled():
    """Test that set_acceleration_at raises error if storage not enabled."""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)

    traj = OrbitTrajectory(6, OrbitFrame.ECI, OrbitRepresentation.CARTESIAN, None)
    # Storage NOT enabled
    traj.add(t0, np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0]))

    acc = np.array([-9.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Acceleration storage is not enabled"):
        traj.set_acceleration_at(0, acc)
