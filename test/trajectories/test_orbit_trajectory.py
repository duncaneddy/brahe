"""
Tests for OrbitTrajectory class in brahe.

These tests provide 1:1 parity with the Rust test suite in src/trajectories/orbit_trajectory.rs
"""

import pytest
import numpy as np
from brahe import (
    Epoch,
    OrbitTrajectory,
    OrbitFrame,
    OrbitRepresentation,
    AngleFormat,
    InterpolationMethod,
    R_EARTH,
    state_osculating_to_cartesian,
    state_cartesian_to_osculating,
    state_eci_to_ecef,
    state_ecef_to_eci,
)
from brahe._brahe import PanicException


def create_test_trajectory():
    """Helper function to create a test trajectory (mirrors Rust helper)."""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.keplerian,
        AngleFormat.degrees,
    )

    epoch1 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    state1 = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
    traj.add_state(epoch1, state1)

    epoch2 = Epoch.from_datetime(2023, 1, 1, 12, 10, 0.0, 0.0, "UTC")
    state2 = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 60.0])
    traj.add_state(epoch2, state2)

    epoch3 = Epoch.from_datetime(2023, 1, 1, 12, 20, 0.0, 0.0, "UTC")
    state3 = np.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 75.0])
    traj.add_state(epoch3, state3)

    return traj


def test_orbittrajectory_new():
    """Rust: test_orbittrajectory_new"""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    assert len(traj) == 0
    assert traj.frame == OrbitFrame.eci
    assert traj.representation == OrbitRepresentation.cartesian
    assert traj.angle_format == AngleFormat.none


def test_orbittrajectory_new_invalid_keplerian_none():
    """Rust: test_orbittrajectory_new_invalid_keplerian_none"""
    with pytest.raises(PanicException, match="Angle format must be specified for Keplerian elements"):
        OrbitTrajectory(
            OrbitFrame.eci,
            OrbitRepresentation.keplerian,
            AngleFormat.none,
        )


def test_orbittrajectory_new_invalid_cartesian_degrees():
    """Rust: test_orbittrajectory_new_invalid_cartesian_degrees"""
    with pytest.raises(PanicException, match="Angle format should be None for Cartesian representation"):
        OrbitTrajectory(
            OrbitFrame.eci,
            OrbitRepresentation.cartesian,
            AngleFormat.degrees,
        )


def test_orbittrajectory_new_invalid_cartesian_radians():
    """Rust: test_orbittrajectory_new_invalid_cartesian_radians"""
    with pytest.raises(PanicException, match="Angle format should be None for Cartesian representation"):
        OrbitTrajectory(
            OrbitFrame.eci,
            OrbitRepresentation.cartesian,
            AngleFormat.radians,
        )


def test_orbittrajectory_new_invalid_keplerian_ecef_degrees():
    """Rust: test_orbittrajectory_new_invalid_keplerian_ecef_degrees"""
    with pytest.raises(PanicException, match="Keplerian elements should be in ECI frame"):
        OrbitTrajectory(
            OrbitFrame.ecef,
            OrbitRepresentation.keplerian,
            AngleFormat.degrees,
        )


def test_orbittrajectory_new_invalid_keplerian_ecef_radians():
    """Rust: test_orbittrajectory_new_invalid_keplerian_ecef_radians"""
    with pytest.raises(PanicException, match="Keplerian elements should be in ECI frame"):
        OrbitTrajectory(
            OrbitFrame.ecef,
            OrbitRepresentation.keplerian,
            AngleFormat.radians,
        )


def test_orbittrajectory_new_invalid_keplerian_ecef_none():
    """Rust: test_orbittrajectory_new_invalid_keplerian_ecef_none"""
    with pytest.raises(PanicException, match="Angle format must be specified for Keplerian elements"):
        OrbitTrajectory(
            OrbitFrame.ecef,
            OrbitRepresentation.keplerian,
            AngleFormat.none,
        )


def test_orbittrajetory_dimension():
    """Rust: test_orbittrajetory_dimension (note the typo!)"""
    traj = create_test_trajectory()
    assert traj.dimension() == 6


def test_orbittrajectory_to_matrix():
    """Rust: test_orbittrajectory_to_matrix"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
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


def test_orbittrajectory_trajectory_add_state():
    """Rust: test_orbittrajectory_trajectory_add_state"""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    # Add states in order
    epoch1 = Epoch.from_jd(2451545.0, "UTC")
    state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add_state(epoch1, state1)

    epoch3 = Epoch.from_jd(2451545.2, "UTC")
    state3 = np.array([7200e3, 0.0, 0.0, 0.0, 7.7e3, 0.0])
    traj.add_state(epoch3, state3)

    # Add a state in between
    epoch2 = Epoch.from_jd(2451545.1, "UTC")
    state2 = np.array([7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0])
    traj.add_state(epoch2, state2)

    assert len(traj) == 3
    assert traj.epoch(0).jd() == 2451545.0
    assert traj.epoch(1).jd() == 2451545.1
    assert traj.epoch(2).jd() == 2451545.2


def test_orbittrajectory_trajectory_state_at_index():
    """Rust: test_orbittrajectory_trajectory_state_at_index"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

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


def test_orbittrajectory_trajectory_epoch_at_index():
    """Rust: test_orbittrajectory_trajectory_epoch_at_index"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

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


def test_orbittrajectory_trajectory_nearest_state():
    """Rust: test_orbittrajectory_trajectory_nearest_state"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    # Test before first epoch
    test_epoch = Epoch.from_jd(2451544.9, "UTC")
    nearest_epoch, nearest_state = traj.nearest_state(test_epoch)
    assert nearest_epoch.jd() == epochs[0].jd()
    assert nearest_state[0] == 7000e3

    # Test after last epoch
    test_epoch = Epoch.from_jd(2451545.3, "UTC")
    nearest_epoch, nearest_state = traj.nearest_state(test_epoch)
    assert nearest_epoch.jd() == epochs[2].jd()
    assert nearest_state[0] == 7200e3

    # Test between epochs
    test_epoch = Epoch.from_jd(2451545.15, "UTC")
    nearest_epoch, nearest_state = traj.nearest_state(test_epoch)
    assert nearest_epoch.jd() == epochs[1].jd()
    assert nearest_state[0] == 7100e3

    # Test exact match
    test_epoch = Epoch.from_jd(2451545.1, "UTC")
    nearest_epoch, nearest_state = traj.nearest_state(test_epoch)
    assert nearest_epoch.jd() == epochs[1].jd()
    assert nearest_state[0] == 7100e3

    # Test just before second epoch
    test_epoch = Epoch.from_jd(2451545.0999, "UTC")
    nearest_epoch, nearest_state = traj.nearest_state(test_epoch)
    assert nearest_epoch.jd() == epochs[1].jd()
    assert nearest_state[0] == 7100e3


def test_orbittrajectory_trajectory_len():
    """Rust: test_orbittrajectory_trajectory_len"""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    assert len(traj) == 0
    assert traj.is_empty()

    epoch = Epoch.from_jd(2451545.0, "UTC")
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add_state(epoch, state)

    assert len(traj) == 1
    assert not traj.is_empty()


def test_orbittrajectory_trajectory_is_empty():
    """Rust: test_orbittrajectory_trajectory_is_empty"""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    assert traj.is_empty()

    epoch = Epoch.from_jd(2451545.0, "UTC")
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add_state(epoch, state)

    assert not traj.is_empty()


def test_orbittrajectory_trajectory_start_epoch():
    """Rust: test_orbittrajectory_trajectory_start_epoch"""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    assert traj.start_epoch() is None

    epoch = Epoch.from_jd(2451545.0, "UTC")
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add_state(epoch, state)

    assert traj.start_epoch().jd() == epoch.jd()


def test_orbittrajectory_trajectory_end_epoch():
    """Rust: test_orbittrajectory_trajectory_end_epoch"""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    assert traj.end_epoch() is None

    epoch1 = Epoch.from_jd(2451545.0, "UTC")
    epoch2 = Epoch.from_jd(2451545.1, "UTC")
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add_state(epoch1, state)
    traj.add_state(epoch2, state)

    assert traj.end_epoch().jd() == epoch2.jd()


def test_orbittrajectory_trajectory_timespan():
    """Rust: test_orbittrajectory_trajectory_timespan"""
    epochs = [
        Epoch.from_jd(2451545.0, "UTC"),
        Epoch.from_jd(2451545.1, "UTC"),
    ]
    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
    ])
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    timespan = traj.timespan()
    assert timespan == pytest.approx(0.1 * 86400.0, abs=1e-5)


def test_orbittrajectory_trajectory_first():
    """Rust: test_orbittrajectory_trajectory_first"""
    epochs = [
        Epoch.from_jd(2451545.0, "UTC"),
        Epoch.from_jd(2451545.1, "UTC"),
    ]
    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
    ])
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    first_epoch, first_state = traj.first()
    assert first_epoch.jd() == epochs[0].jd()
    assert np.array_equal(first_state, np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]))


def test_orbittrajectory_trajectory_last():
    """Rust: test_orbittrajectory_trajectory_last"""
    epochs = [
        Epoch.from_jd(2451545.0, "UTC"),
        Epoch.from_jd(2451545.1, "UTC"),
    ]
    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
    ])
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    last_epoch, last_state = traj.last()
    assert last_epoch.jd() == epochs[1].jd()
    assert np.array_equal(last_state, np.array([7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0]))


def test_orbittrajectory_trajectory_clear():
    """Rust: test_orbittrajectory_trajectory_clear"""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    epoch = Epoch.from_jd(2451545.0, "UTC")
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add_state(epoch, state)

    assert len(traj) == 1
    traj.clear()
    assert len(traj) == 0


def test_orbittrajectory_trajectory_remove_state():
    """Rust: test_orbittrajectory_trajectory_remove_state"""
    epochs = [
        Epoch.from_jd(2451545.0, "UTC"),
        Epoch.from_jd(2451545.1, "UTC"),
    ]
    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
    ])
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    removed_state = traj.remove_state(epochs[0])
    assert removed_state[0] == 7000e3
    assert len(traj) == 1


def test_orbittrajectory_trajectory_remove_state_at_index():
    """Rust: test_orbittrajectory_trajectory_remove_state_at_index"""
    epochs = [
        Epoch.from_jd(2451545.0, "UTC"),
        Epoch.from_jd(2451545.1, "UTC"),
    ]
    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
    ])
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    removed_epoch, removed_state = traj.remove_state_at_index(0)
    assert removed_epoch.jd() == 2451545.0
    assert removed_state[0] == 7000e3
    assert len(traj) == 1


def test_orbittrajectory_trajectory_get():
    """Rust: test_orbittrajectory_trajectory_get"""
    epochs = [
        Epoch.from_jd(2451545.0, "UTC"),
        Epoch.from_jd(2451545.1, "UTC"),
    ]
    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
    ])
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    epoch, state = traj.get(1)
    assert epoch.jd() == 2451545.1
    assert state[0] == 7100e3


def test_orbittrajectory_trajectory_index_before_epoch():
    """Rust: test_orbittrajectory_trajectory_index_before_epoch"""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
    ])

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
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
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
    ])

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
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
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
    ])

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
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
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
    ])

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
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
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    # Add 5 states
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    for i in range(5):
        epoch = t0 + (i * 60.0)
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add_state(epoch, state)

    assert len(traj) == 5

    # Set max size to 3
    traj.set_eviction_policy_max_size(3)

    # Should only have 3 most recent states
    assert len(traj) == 3

    # First state should be the 3rd original state (oldest 2 evicted)
    first_state = traj.state(0)
    assert first_state[0] == pytest.approx(7000e3 + 2000.0, abs=1.0)

    # Add another state - should still maintain max size
    new_epoch = t0 + 5.0 * 60.0
    new_state = np.array([7000e3 + 5000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj.add_state(new_epoch, new_state)

    assert len(traj) == 3

    # Test error case
    with pytest.raises(Exception):
        traj.set_eviction_policy_max_size(0)


def test_orbittrajectory_trajectory_set_eviction_policy_max_age():
    """Rust: test_orbittrajectory_trajectory_set_eviction_policy_max_age"""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    # Add states spanning 5 minutes
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    for i in range(6):
        epoch = t0 + (i * 60.0)  # 0, 60, 120, 180, 240, 300 seconds
        state = np.array([7000e3 + i * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        traj.add_state(epoch, state)

    assert len(traj) == 6

    # Set max age to 240 seconds
    traj.set_eviction_policy_max_age(240.0)
    assert len(traj) == 5

    first_state = traj.state(0)
    assert first_state[0] == pytest.approx(7000e3 + 1000.0, abs=1.0)

    # Set max age to 239 seconds
    traj.set_eviction_policy_max_age(239.0)

    assert len(traj) == 4
    first_state = traj.state(0)
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
    assert traj.frame == OrbitFrame.eci
    assert traj.representation == OrbitRepresentation.cartesian
    assert traj.angle_format == AngleFormat.none


def test_orbittrajectory_index_index():
    """Rust: test_orbittrajectory_index_index"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
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
    epochs = [Epoch.from_jd(2451545.0, "UTC")]
    states = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    with pytest.raises(Exception):
        _ = traj[10]


def test_orbittrajectory_intoiterator_into_iter():
    """Rust: test_orbittrajectory_intoiterator_into_iter"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
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
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    count = 0
    for _ in traj:
        count += 1
    assert count == 0


def test_orbittrajectory_iterator_iterator_size_hint():
    """Rust: test_orbittrajectory_iterator_iterator_size_hint"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    # Python's iterator doesn't have size_hint, but we can use len() on trajectory
    iter_obj = iter(traj)
    assert len(traj) == 3


def test_orbittrajectory_iterator_iterator_len():
    """Rust: test_orbittrajectory_iterator_iterator_len"""
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
    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    iter_obj = iter(traj)
    assert len(traj) == 3


def test_orbittrajectory_interpolatable_set_interpolation_method():
    """Rust: test_orbittrajectory_interpolatable_set_interpolation_method"""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    assert traj.get_interpolation_method() == InterpolationMethod.linear

    traj.set_interpolation_method(InterpolationMethod.linear)
    assert traj.get_interpolation_method() == InterpolationMethod.linear


def test_orbittrajectory_interpolatable_get_interpolation_method():
    """Rust: test_orbittrajectory_interpolatable_get_interpolation_method"""
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    # Test that get_interpolation_method returns Linear
    assert traj.get_interpolation_method() == InterpolationMethod.linear

    # Set it to different methods and verify get_interpolation_method returns the correct value
    traj.set_interpolation_method(InterpolationMethod.linear)
    assert traj.get_interpolation_method() == InterpolationMethod.linear


def test_orbittrajectory_interpolatable_interpolate_linear():
    """Rust: test_orbittrajectory_interpolatable_interpolate_linear"""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        60.0, 120.0, 180.0, 240.0, 300.0, 360.0,
        120.0, 240.0, 360.0, 480.0, 600.0, 720.0,
    ])

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
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
    single_state = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    single_traj = OrbitTrajectory.from_orbital_data(
        single_epoch,
        single_state,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    state_single = single_traj.interpolate_linear(t0)
    assert state_single[0] == pytest.approx(100.0, abs=1e-10)
    assert state_single[1] == pytest.approx(200.0, abs=1e-10)


def test_orbittrajectory_interpolatable_interpolate():
    """Rust: test_orbittrajectory_interpolatable_interpolate"""
    t0 = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    t1 = t0 + 60.0
    t2 = t0 + 120.0

    epochs = [t0, t1, t2]
    states = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        60.0, 120.0, 180.0, 240.0, 300.0, 360.0,
        120.0, 240.0, 360.0, 480.0, 600.0, 720.0,
    ])

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    # Test that interpolate() with Linear method returns same result as interpolate_linear()
    t0_plus_30 = t0 + 30.0
    state_interpolate = traj.interpolate(t0_plus_30)
    state_interpolate_linear = traj.interpolate_linear(t0_plus_30)

    for i in range(6):
        assert state_interpolate[i] == pytest.approx(state_interpolate_linear[i], abs=1e-10)


def test_orbittrajectory_orbitaltrajectory_from_orbital_data():
    """Rust: test_orbittrajectory_orbitaltrajectory_from_orbital_data"""
    epochs = [
        Epoch.from_jd(2451545.0, "UTC"),
        Epoch.from_jd(2451545.1, "UTC"),
    ]
    states = np.array([
        7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0,
        7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0,
    ])

    traj = OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    assert len(traj) == 2
    assert traj.frame == OrbitFrame.eci
    assert traj.representation == OrbitRepresentation.cartesian


def test_orbittrajectory_orbitaltrajectory_to_eci():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_eci"""
    tol = 1e-6

    state_base = state_osculating_to_cartesian(
        np.array([R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0]),
        True,
    )

    # No transformation needed if already in ECI
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    traj.add_state(epoch, state_base)

    eci_traj = traj.to_eci()
    assert eci_traj.frame == OrbitFrame.eci
    assert eci_traj.representation == OrbitRepresentation.cartesian
    assert eci_traj.angle_format == AngleFormat.none
    assert len(eci_traj) == 1
    epoch_out, state_out = eci_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to ECI - Radians
    kep_traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.keplerian,
        AngleFormat.radians,
    )
    kep_state_rad = state_cartesian_to_osculating(state_base, False)
    kep_traj.add_state(epoch, kep_state_rad)

    eci_from_kep_rad = kep_traj.to_eci()
    assert eci_from_kep_rad.frame == OrbitFrame.eci
    assert eci_from_kep_rad.representation == OrbitRepresentation.cartesian
    assert eci_from_kep_rad.angle_format == AngleFormat.none
    assert len(eci_from_kep_rad) == 1
    epoch_out, state_out = eci_from_kep_rad.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to ECI - Degrees
    kep_traj_deg = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.keplerian,
        AngleFormat.degrees,
    )
    kep_state_deg = state_cartesian_to_osculating(state_base, True)
    kep_traj_deg.add_state(epoch, kep_state_deg)
    eci_from_kep_deg = kep_traj_deg.to_eci()
    assert eci_from_kep_deg.frame == OrbitFrame.eci
    assert eci_from_kep_deg.representation == OrbitRepresentation.cartesian
    assert eci_from_kep_deg.angle_format == AngleFormat.none
    assert len(eci_from_kep_deg) == 1
    epoch_out, state_out = eci_from_kep_deg.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert ECEF to ECI
    ecef_traj = OrbitTrajectory(
        OrbitFrame.ecef,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )
    ecef_state = state_eci_to_ecef(epoch, state_base)
    ecef_traj.add_state(epoch, ecef_state)
    eci_from_ecef = ecef_traj.to_eci()
    assert eci_from_ecef.frame == OrbitFrame.eci
    assert eci_from_ecef.representation == OrbitRepresentation.cartesian
    assert eci_from_ecef.angle_format == AngleFormat.none
    assert len(eci_from_ecef) == 1
    epoch_out, state_out = eci_from_ecef.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)


def test_orbittrajectory_orbitaltrajectory_to_ecef():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_ecef"""
    tol = 1e-6

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    state_base = state_eci_to_ecef(
        epoch,
        state_osculating_to_cartesian(
            np.array([R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0]),
            True,
        ),
    )

    # No transformation needed if already in ECEF
    traj = OrbitTrajectory(
        OrbitFrame.ecef,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )

    traj.add_state(epoch, state_base)
    ecef_traj = traj.to_ecef()
    assert ecef_traj.frame == OrbitFrame.ecef
    assert ecef_traj.representation == OrbitRepresentation.cartesian
    assert ecef_traj.angle_format == AngleFormat.none
    assert len(ecef_traj) == 1
    epoch_out, state_out = ecef_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert ECI to ECEF
    eci_traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )
    eci_state = state_ecef_to_eci(epoch, state_base)
    eci_traj.add_state(epoch, eci_state)
    ecef_from_eci = eci_traj.to_ecef()
    assert ecef_from_eci.frame == OrbitFrame.ecef
    assert ecef_from_eci.representation == OrbitRepresentation.cartesian
    assert ecef_from_eci.angle_format == AngleFormat.none
    assert len(ecef_from_eci) == 1
    epoch_out, state_out = ecef_from_eci.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to ECEF - Radians
    kep_traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.keplerian,
        AngleFormat.radians,
    )
    kep_state_rad = state_cartesian_to_osculating(eci_state, False)
    kep_traj.add_state(epoch, kep_state_rad)
    ecef_from_kep_rad = kep_traj.to_ecef()
    assert ecef_from_kep_rad.frame == OrbitFrame.ecef
    assert ecef_from_kep_rad.representation == OrbitRepresentation.cartesian
    assert ecef_from_kep_rad.angle_format == AngleFormat.none
    assert len(ecef_from_kep_rad) == 1
    epoch_out, state_out = ecef_from_kep_rad.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)

    # Convert Keplerian to ECEF - Degrees
    kep_traj_deg = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.keplerian,
        AngleFormat.degrees,
    )
    kep_state_deg = state_cartesian_to_osculating(eci_state, True)
    kep_traj_deg.add_state(epoch, kep_state_deg)
    ecef_from_kep_deg = kep_traj_deg.to_ecef()
    assert ecef_from_kep_deg.frame == OrbitFrame.ecef
    assert ecef_from_kep_deg.representation == OrbitRepresentation.cartesian
    assert ecef_from_kep_deg.angle_format == AngleFormat.none
    assert len(ecef_from_kep_deg) == 1
    epoch_out, state_out = ecef_from_kep_deg.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_base[i], abs=tol)


def test_orbittrajectory_orbitaltrajectory_to_keplerian_deg():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_keplerian_deg"""
    tol = 1e-6
    DEG2RAD = 0.017453292519943295

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    state_kep_deg = np.array([7000e3, 0.01, 97.0, 15.0, 30.0, 45.0])

    # No transformation needed if already in Keplerian Degrees
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.keplerian,
        AngleFormat.degrees,
    )
    traj.add_state(epoch, state_kep_deg)
    kep_traj = traj.to_keplerian(AngleFormat.degrees)
    assert kep_traj.frame == OrbitFrame.eci
    assert kep_traj.representation == OrbitRepresentation.keplerian
    assert kep_traj.angle_format == AngleFormat.degrees
    assert len(kep_traj) == 1
    epoch_out, state_out = kep_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_deg[i], abs=tol)

    # Convert Keplerian Radians to Keplerian Degrees
    kep_rad_traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.keplerian,
        AngleFormat.radians,
    )
    state_kep_rad = state_kep_deg.copy()
    for i in range(2, 6):
        state_kep_rad[i] = state_kep_deg[i] * DEG2RAD
    kep_rad_traj.add_state(epoch, state_kep_rad)
    kep_from_rad = kep_rad_traj.to_keplerian(AngleFormat.degrees)
    assert kep_from_rad.frame == OrbitFrame.eci
    assert kep_from_rad.representation == OrbitRepresentation.keplerian
    assert kep_from_rad.angle_format == AngleFormat.degrees
    assert len(kep_from_rad) == 1
    epoch_out, state_out = kep_from_rad.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_deg[i], abs=tol)

    # Convert ECI to Keplerian Degrees
    cart_traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )
    cart_state = state_osculating_to_cartesian(state_kep_deg, True)
    cart_traj.add_state(epoch, cart_state)
    kep_from_cart = cart_traj.to_keplerian(AngleFormat.degrees)
    assert kep_from_cart.frame == OrbitFrame.eci
    assert kep_from_cart.representation == OrbitRepresentation.keplerian
    assert kep_from_cart.angle_format == AngleFormat.degrees
    assert len(kep_from_cart) == 1
    epoch_out, state_out = kep_from_cart.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_deg[i], abs=tol)

    # Convert ECEF to Keplerian Degrees
    ecef_traj = OrbitTrajectory(
        OrbitFrame.ecef,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )
    ecef_state = state_eci_to_ecef(epoch, cart_state)
    ecef_traj.add_state(epoch, ecef_state)
    kep_from_ecef = ecef_traj.to_keplerian(AngleFormat.degrees)
    assert kep_from_ecef.frame == OrbitFrame.eci
    assert kep_from_ecef.representation == OrbitRepresentation.keplerian
    assert kep_from_ecef.angle_format == AngleFormat.degrees
    assert len(kep_from_ecef) == 1
    epoch_out, state_out = kep_from_ecef.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_deg[i], abs=tol)


def test_orbittrajectory_orbitaltrajectory_to_keplerian_rad():
    """Rust: test_orbittrajectory_orbitaltrajectory_to_keplerian_rad"""
    tol = 1e-6
    DEG2RAD = 0.017453292519943295

    epoch = Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    state_kep_deg = np.array([7000e3, 0.01, 97.0, 15.0, 30.0, 45.0])
    state_kep_rad = state_kep_deg.copy()
    for i in range(2, 6):
        state_kep_rad[i] = state_kep_deg[i] * DEG2RAD

    # No transformation needed if already in Keplerian Radians
    traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.keplerian,
        AngleFormat.radians,
    )
    traj.add_state(epoch, state_kep_rad)
    kep_traj = traj.to_keplerian(AngleFormat.radians)
    assert kep_traj.frame == OrbitFrame.eci
    assert kep_traj.representation == OrbitRepresentation.keplerian
    assert kep_traj.angle_format == AngleFormat.radians
    assert len(kep_traj) == 1
    epoch_out, state_out = kep_traj.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_rad[i], abs=tol)

    # Convert Keplerian Degrees to Keplerian Radians
    kep_deg_traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.keplerian,
        AngleFormat.degrees,
    )
    kep_deg_traj.add_state(epoch, state_kep_deg)
    kep_from_deg = kep_deg_traj.to_keplerian(AngleFormat.radians)
    assert kep_from_deg.frame == OrbitFrame.eci
    assert kep_from_deg.representation == OrbitRepresentation.keplerian
    assert kep_from_deg.angle_format == AngleFormat.radians
    assert len(kep_from_deg) == 1
    epoch_out, state_out = kep_from_deg.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_rad[i], abs=tol)

    # Convert ECI to Keplerian Radians
    cart_traj = OrbitTrajectory(
        OrbitFrame.eci,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )
    cart_state = state_osculating_to_cartesian(state_kep_deg, True)
    cart_traj.add_state(epoch, cart_state)
    kep_from_cart = cart_traj.to_keplerian(AngleFormat.radians)
    assert kep_from_cart.frame == OrbitFrame.eci
    assert kep_from_cart.representation == OrbitRepresentation.keplerian
    assert kep_from_cart.angle_format == AngleFormat.radians
    assert len(kep_from_cart) == 1
    epoch_out, state_out = kep_from_cart.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_rad[i], abs=tol)

    # Convert ECEF to Keplerian Radians
    ecef_traj = OrbitTrajectory(
        OrbitFrame.ecef,
        OrbitRepresentation.cartesian,
        AngleFormat.none,
    )
    ecef_state = state_eci_to_ecef(epoch, cart_state)
    ecef_traj.add_state(epoch, ecef_state)
    kep_from_ecef = ecef_traj.to_keplerian(AngleFormat.radians)
    assert kep_from_ecef.frame == OrbitFrame.eci
    assert kep_from_ecef.representation == OrbitRepresentation.keplerian
    assert kep_from_ecef.angle_format == AngleFormat.radians
    assert len(kep_from_ecef) == 1
    epoch_out, state_out = kep_from_ecef.get(0)
    assert epoch_out.jd() == epoch.jd()
    for i in range(6):
        assert state_out[i] == pytest.approx(state_kep_rad[i], abs=tol)
