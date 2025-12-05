"""
Tests for propagator utility functions Python bindings

These tests mirror the Rust tests from src/propagators/functions.rs
"""

import numpy as np
import pytest
from brahe import (
    Epoch,
    TimeSystem,
    KeplerianPropagator,
    SGPPropagator,
    AngleFormat,
    state_koe_to_eci,
    par_propagate_to,
)


def test_par_propagate_to_keplerian():
    """Test parallel propagation of Keplerian propagators"""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem.UTC)
    target = epoch + 3600.0  # 1 hour later

    # Create multiple Keplerian propagators with different initial conditions
    propagators = []

    oe1 = np.array([7000e3, 0.001, 98.0, 0.0, 0.0, 0.0])
    state1 = state_koe_to_eci(oe1, AngleFormat.DEGREES)
    prop1 = KeplerianPropagator.from_eci(epoch, state1, 60.0)
    propagators.append(prop1)

    oe2 = np.array([7200e3, 0.002, 97.0, 10.0, 20.0, 30.0])
    state2 = state_koe_to_eci(oe2, AngleFormat.DEGREES)
    prop2 = KeplerianPropagator.from_eci(epoch, state2, 60.0)
    propagators.append(prop2)

    oe3 = np.array([6800e3, 0.0005, 51.6, 45.0, 90.0, 120.0])
    state3 = state_koe_to_eci(oe3, AngleFormat.DEGREES)
    prop3 = KeplerianPropagator.from_eci(epoch, state3, 60.0)
    propagators.append(prop3)

    # Propagate in parallel
    par_propagate_to(propagators, target)

    # Verify all propagators reached target epoch
    for prop in propagators:
        assert prop.current_epoch == target

    # Verify states are different (they had different initial conditions)
    state0 = prop1.current_state()
    state1_result = prop2.current_state()
    state2_result = prop3.current_state()

    assert state0[0] != state1_result[0]
    assert state0[0] != state2_result[0]
    assert state1_result[0] != state2_result[0]


def test_par_propagate_to_sgp():
    """Test parallel propagation of SGP propagators"""
    # ISS TLE data (using same TLE multiple times to test parallel execution)
    line1_iss = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2_iss = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    epoch_iss = SGPPropagator.from_tle(line1_iss, line2_iss, 60.0).epoch

    # Create multiple propagators from same TLE
    propagators = [
        SGPPropagator.from_tle(line1_iss, line2_iss, 60.0),
        SGPPropagator.from_tle(line1_iss, line2_iss, 60.0),
        SGPPropagator.from_tle(line1_iss, line2_iss, 60.0),
    ]

    # Propagate all forward 1 hour from TLE epoch
    target = epoch_iss + 3600.0
    par_propagate_to(propagators, target)

    # Verify all reached target epoch
    for prop in propagators:
        assert prop.current_epoch == target

    # Verify states are the same (same TLE, same propagation)
    state0 = propagators[0].current_state()
    state1 = propagators[1].current_state()
    state2 = propagators[2].current_state()

    for i in range(6):
        assert abs(state0[i] - state1[i]) < 1e-9
        assert abs(state0[i] - state2[i]) < 1e-9


def test_par_propagate_to_matches_sequential():
    """Test that parallel propagation gives same results as sequential"""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem.UTC)
    target = epoch + 7200.0  # 2 hours

    # Create identical propagators for parallel test
    oe1 = np.array([7000e3, 0.001, 98.0, 0.0, 0.0, 0.0])
    state1 = state_koe_to_eci(oe1, AngleFormat.DEGREES)

    oe2 = np.array([7200e3, 0.002, 97.0, 10.0, 20.0, 30.0])
    state2 = state_koe_to_eci(oe2, AngleFormat.DEGREES)

    parallel_props = [
        KeplerianPropagator.from_eci(epoch, state1, 60.0),
        KeplerianPropagator.from_eci(epoch, state2, 60.0),
    ]

    sequential_props = [
        KeplerianPropagator.from_eci(epoch, state1, 60.0),
        KeplerianPropagator.from_eci(epoch, state2, 60.0),
    ]

    # Propagate in parallel
    par_propagate_to(parallel_props, target)

    # Propagate sequentially
    for prop in sequential_props:
        prop.propagate_to(target)

    # Results should be identical
    for i in range(len(parallel_props)):
        assert parallel_props[i].current_epoch == sequential_props[i].current_epoch

        parallel_state = parallel_props[i].current_state()
        sequential_state = sequential_props[i].current_state()

        for j in range(6):
            assert abs(parallel_state[j] - sequential_state[j]) < 1e-9


def test_par_propagate_to_empty_list():
    """Test that empty list doesn't cause errors"""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem.UTC)
    target = epoch + 3600.0

    propagators = []

    # Should not raise an error with empty list
    par_propagate_to(propagators, target)


def test_par_propagate_to_single_propagator():
    """Test parallel propagation with single propagator"""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem.UTC)
    target = epoch + 3600.0

    oe = np.array([7000e3, 0.001, 98.0, 0.0, 0.0, 0.0])
    state = state_koe_to_eci(oe, AngleFormat.DEGREES)

    propagators = [KeplerianPropagator.from_eci(epoch, state, 60.0)]

    par_propagate_to(propagators, target)

    assert propagators[0].current_epoch == target


def test_par_propagate_to_mixed_types_raises_error():
    """Test that mixing propagator types raises an error"""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem.UTC)
    target = epoch + 3600.0

    oe = np.array([7000e3, 0.001, 98.0, 0.0, 0.0, 0.0])
    state = state_koe_to_eci(oe, AngleFormat.DEGREES)
    kep_prop = KeplerianPropagator.from_eci(epoch, state, 60.0)

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    sgp_prop = SGPPropagator.from_tle(line1, line2, 60.0)

    # Create mixed list - first is Keplerian, second is SGP
    # This should raise a TypeError because types don't match
    propagators = [kep_prop, sgp_prop]

    with pytest.raises(TypeError):
        par_propagate_to(propagators, target)


def test_par_propagate_to_not_a_list_raises_error():
    """Test that passing non-list raises an error"""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem.UTC)
    target = epoch + 3600.0

    oe = np.array([7000e3, 0.001, 98.0, 0.0, 0.0, 0.0])
    state = state_koe_to_eci(oe, AngleFormat.DEGREES)
    prop = KeplerianPropagator.from_eci(epoch, state, 60.0)

    # Pass a single propagator instead of a list
    with pytest.raises(TypeError):
        par_propagate_to(prop, target)


def test_par_propagate_to_sgp_with_events():
    """Test that parallel propagation detects events correctly for SGP propagators"""
    import brahe as bh

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    # Create multiple SGP propagators from the same TLE
    propagators = [SGPPropagator.from_tle(line1, line2, 60.0) for _ in range(3)]
    epoch = propagators[0].epoch

    # Add time events to each propagator at different times
    for i, prop in enumerate(propagators):
        event = bh.TimeEvent(epoch + 100.0 * (i + 1), f"Event_{i}")
        prop.add_event_detector(event)

    # Propagate in parallel
    target = epoch + 400.0
    par_propagate_to(propagators, target)

    # Verify all events were detected
    for i, prop in enumerate(propagators):
        event_log = prop.event_log()
        assert len(event_log) == 1, (
            f"Propagator {i} should have 1 event, got {len(event_log)}"
        )
        assert f"Event_{i}" in event_log[0].name, f"Event name should contain Event_{i}"


def test_par_propagate_to_numerical_propagator_raises_error():
    """Test that NumericalPropagator raises a clear error due to GIL limitations"""
    import brahe as bh

    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem.UTC)
    target = epoch + 6.0
    state = np.array([1.0, 0.0])

    # Simple harmonic oscillator dynamics
    omega = 1.0

    def sho_dynamics(t, state, params):
        return np.array([state[1], -(omega**2) * state[0]])

    config = bh.NumericalPropagationConfig.default()

    propagators = [
        bh.NumericalPropagator(epoch, state.copy(), sho_dynamics, config),
        bh.NumericalPropagator(epoch, state.copy(), sho_dynamics, config),
    ]

    # Should raise TypeError with helpful message about GIL
    with pytest.raises(TypeError, match="GIL"):
        par_propagate_to(propagators, target)
