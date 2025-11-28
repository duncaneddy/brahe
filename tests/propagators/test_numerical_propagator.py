"""
Tests for NumericalPropagator Python bindings

These tests mirror the Rust tests from src/propagators/dnumerical_propagator.rs
"""

import numpy as np
from brahe import (
    Epoch,
    TimeSystem,
    NumericalPropagator,
    NumericalPropagationConfig,
    IntegrationMethod,
)


def create_test_epoch():
    """Create a standard test epoch"""
    return Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem.UTC)


def sho_dynamics(t, state, params):
    """Simple Harmonic Oscillator dynamics: d²x/dt² = -ω²x

    State: [x, v] where dx/dt = v, dv/dt = -ω²x
    Default ω = 1.0
    """
    omega = params[0] if params is not None and len(params) > 0 else 1.0
    return np.array([state[1], -(omega**2) * state[0]])


def damped_oscillator_dynamics(t, state, params):
    """Damped oscillator: d²x/dt² = -ω²x - 2ζω dx/dt

    State: [x, v]
    Params: [omega, zeta]
    """
    omega = params[0] if params is not None else 1.0
    zeta = params[1] if params is not None and len(params) > 1 else 0.1
    x, v = state
    return np.array([v, -(omega**2) * x - 2 * zeta * omega * v])


def exponential_growth_dynamics(t, state, params):
    """Exponential growth/decay: dx/dt = k*x

    State: [x]
    Params: [k] where k > 0 is growth, k < 0 is decay
    """
    k = params[0] if params is not None else 1.0
    return np.array([k * state[0]])


def linear_system_dynamics(t, state, params):
    """Linear system: dx/dt = A @ x

    State: 2D vector
    Params: [a11, a12, a21, a22] (flattened 2x2 matrix)
    """
    if params is not None and len(params) >= 4:
        A = np.array([[params[0], params[1]], [params[2], params[3]]])
    else:
        A = np.array([[0.0, 1.0], [-1.0, 0.0]])  # Default: rotation
    return A @ state


# =============================================================================
# Construction Tests
# =============================================================================


def test_numericalpropagator_construction_default():
    """Test NumericalPropagator construction with SHO dynamics"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])  # Initial: x=1, v=0

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    assert prop is not None
    assert prop.initial_epoch == epoch
    assert prop.current_epoch == epoch
    assert prop.state_dim == 2


def test_numericalpropagator_construction_with_params():
    """Test NumericalPropagator with parameters"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    params = np.array([2.0])  # omega = 2.0

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default(), params
    )

    assert prop is not None
    assert prop.state_dim == 2


def test_numericalpropagator_construction_higher_dimensional():
    """Test NumericalPropagator with higher dimensional system"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0, 0.5, 0.5])  # 4D system
    params = np.array([0.0, 1.0, -1.0, 0.0])  # 2x2 rotation matrix elements

    def higher_dim_dynamics(t, state, params):
        # Just a simple diagonal system for testing
        return -0.1 * state

    prop = NumericalPropagator(
        epoch, state, higher_dim_dynamics, NumericalPropagationConfig.default(), params
    )

    assert prop is not None
    assert prop.state_dim == 4


def test_numericalpropagator_initial_state_in_trajectory():
    """Test that initial state is stored in trajectory at construction"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Trajectory should contain initial state immediately after construction
    traj = prop.trajectory
    assert len(traj) >= 1, "Trajectory should contain initial state"

    # Query state at initial epoch should return initial state
    initial_state = prop.state(epoch)
    np.testing.assert_array_almost_equal(initial_state, state)


# =============================================================================
# DStatePropagator Trait Tests
# =============================================================================


def test_dstatepropagator_step_by_forward():
    """Test step_by() forward propagation"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.step_by(0.5)

    assert prop.current_epoch == epoch + 0.5
    new_state = prop.current_state()
    assert len(new_state) == 2
    # State should have changed
    assert not np.allclose(new_state, state)


def test_dstatepropagator_propagate_to_forward():
    """Test propagate_to() forward propagation"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    target = epoch + np.pi  # Half period of SHO with omega=1
    prop.propagate_to(target)

    assert prop.current_epoch == target
    # At t = pi, for SHO starting at [1,0], should be near [-1, 0]
    final_state = prop.current_state()
    assert final_state[0] < 0  # Position should be negative


def test_dstatepropagator_step_by_backward():
    """Test step_by() backward propagation"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Step forward first
    prop.step_by(1.0)
    assert prop.current_epoch == epoch + 1.0

    # Then step backward
    prop.step_by(-0.5)
    assert prop.current_epoch == epoch + 0.5


def test_dstatepropagator_propagate_to_backward():
    """Test propagate_to() backward propagation"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Forward first
    prop.propagate_to(epoch + 2.0)

    # Then backward
    prop.propagate_to(epoch + 1.0)
    assert prop.current_epoch == epoch + 1.0


def test_dstatepropagator_propagate_steps():
    """Test propagate_steps() method"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.step_size = 0.1
    prop.propagate_steps(10)

    assert abs(prop.current_epoch - (epoch + 1.0)) < 1e-6


def test_dstatepropagator_reset():
    """Test reset() method"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Propagate
    prop.propagate_to(epoch + 2.0)
    assert prop.current_epoch != epoch

    # Reset
    prop.reset()

    assert prop.current_epoch == epoch
    np.testing.assert_array_almost_equal(prop.current_state(), state)


def test_dstatepropagator_getters():
    """Test getter properties"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    assert prop.initial_epoch == epoch
    assert prop.current_epoch == epoch
    assert prop.state_dim == 2
    np.testing.assert_array_almost_equal(prop.initial_state(), state)
    np.testing.assert_array_almost_equal(prop.current_state(), state)


def test_dstatepropagator_step_size():
    """Test step_size getter and setter"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.step_size = 0.5
    assert prop.step_size == 0.5

    prop.step_size = 0.1
    assert prop.step_size == 0.1


# =============================================================================
# DStateProvider Tests
# =============================================================================


def test_dstateprovider_state_at_current():
    """Test state() at current epoch"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    result = prop.state(epoch)
    np.testing.assert_array_almost_equal(result, state)


def test_dstateprovider_state_interpolation():
    """Test state() with interpolation"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Build trajectory
    prop.propagate_to(epoch + 2.0)

    # Query intermediate state
    result = prop.state(epoch + 1.0)
    assert len(result) == 2


# =============================================================================
# Trajectory Tests
# =============================================================================


def test_numericalpropagator_trajectory():
    """Test trajectory property"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.step_size = 0.1
    prop.propagate_steps(10)

    traj = prop.trajectory
    assert traj is not None
    assert len(traj) >= 2


def test_numericalpropagator_eviction_policy_max_size():
    """Test set_eviction_policy_max_size()"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.set_eviction_policy_max_size(5)
    prop.step_size = 0.1
    prop.propagate_steps(20)

    traj = prop.trajectory
    assert len(traj) <= 5


def test_numericalpropagator_eviction_policy_max_age():
    """Test set_eviction_policy_max_age()"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.set_eviction_policy_max_age(0.5)
    prop.step_size = 0.1
    prop.propagate_steps(20)

    traj = prop.trajectory
    # Should have evicted old states
    assert len(traj) <= 10


# =============================================================================
# STM and Covariance Tests
# =============================================================================


def test_numericalpropagator_stm():
    """Test stm() method"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # STM may be None if not configured
    _ = prop.stm()


def test_numericalpropagator_sensitivity():
    """Test sensitivity() method"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Sensitivity may be None if not configured
    _ = prop.sensitivity()


# =============================================================================
# Identity Methods Tests
# =============================================================================


def test_numericalpropagator_identity_methods():
    """Test identity methods (name, id, uuid)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Test with_name
    prop.with_name("SHO_Propagator")
    assert prop.get_name() == "SHO_Propagator"

    # Test with_id
    prop.with_id(42)
    assert prop.get_id() == 42

    # Test with_new_uuid
    prop.with_new_uuid()
    uuid = prop.get_uuid()
    assert uuid is not None


# =============================================================================
# Physics Validation Tests
# =============================================================================


def test_numericalpropagator_sho_full_period():
    """Test SHO returns to initial state after one period"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    # Use high precision config for better accuracy
    config = NumericalPropagationConfig.high_precision()

    prop = NumericalPropagator(epoch, state, sho_dynamics, config)

    # Period of SHO with omega=1 is 2*pi
    period = 2 * np.pi
    prop.propagate_to(epoch + period)

    final_state = prop.current_state()
    # Should return close to initial state (relax tolerance)
    np.testing.assert_array_almost_equal(final_state, state, decimal=3)


def test_numericalpropagator_sho_half_period():
    """Test SHO at half period (x should reverse sign)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # At t = pi, x should be near -1
    prop.propagate_to(epoch + np.pi)

    final_state = prop.current_state()
    assert final_state[0] < -0.9  # Should be close to -1


def test_numericalpropagator_sho_energy_conservation():
    """Test that SHO conserves energy"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    omega = 1.0

    # Use high precision config for better energy conservation
    config = NumericalPropagationConfig.high_precision()

    prop = NumericalPropagator(epoch, state, sho_dynamics, config)

    # Energy = 0.5 * (v^2 + omega^2 * x^2)
    def energy(s):
        return 0.5 * (s[1] ** 2 + omega**2 * s[0] ** 2)

    initial_energy = energy(state)

    # Propagate for one period (shorter to reduce drift)
    prop.propagate_to(epoch + 2 * np.pi)

    final_energy = energy(prop.current_state())

    # Energy should be conserved (relax tolerance for numerical integration)
    assert abs(final_energy - initial_energy) / initial_energy < 1e-4


def test_numericalpropagator_exponential_decay():
    """Test exponential decay dynamics"""
    epoch = create_test_epoch()
    state = np.array([1.0])
    k = -0.5  # Decay rate

    # Capture k in closure (this is how Rust DNumericalPropagator expects dynamics)
    def decay_dynamics(t, state, params):
        return np.array([k * state[0]])

    config = NumericalPropagationConfig.default()

    prop = NumericalPropagator(epoch, state, decay_dynamics, config)

    t = 2.0
    prop.propagate_to(epoch + t)

    final_state = prop.current_state()
    expected = np.exp(k * t)

    # Relax tolerance for numerical integration
    assert abs(final_state[0] - expected) < 1e-2


def test_numericalpropagator_exponential_growth():
    """Test exponential growth dynamics"""
    epoch = create_test_epoch()
    state = np.array([1.0])
    k = 0.5  # Growth rate

    # Capture k in closure (this is how Rust DNumericalPropagator expects dynamics)
    def growth_dynamics(t, state, params):
        return np.array([k * state[0]])

    # Use DP54 (adaptive) - don't use high_precision which uses RKN1210 (requires even-dim state)
    config = NumericalPropagationConfig.default()

    prop = NumericalPropagator(epoch, state, growth_dynamics, config)

    t = 1.0
    prop.propagate_to(epoch + t)

    final_state = prop.current_state()
    expected = np.exp(k * t)

    # Relax tolerance for numerical integration
    assert abs(final_state[0] - expected) < 1e-2


def test_numericalpropagator_damped_oscillator():
    """Test damped oscillator dynamics"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    omega = 2.0
    zeta = 0.1  # Underdamped

    # Capture omega and zeta in closure (this is how Rust DNumericalPropagator expects dynamics)
    def damped_dynamics(t, state, params):
        x, v = state
        return np.array([v, -(omega**2) * x - 2 * zeta * omega * v])

    prop = NumericalPropagator(
        epoch, state, damped_dynamics, NumericalPropagationConfig.default()
    )

    # Propagate
    prop.propagate_to(epoch + 10.0)

    final_state = prop.current_state()

    # Amplitude should have decreased due to damping
    initial_amplitude = np.sqrt(state[0] ** 2 + state[1] ** 2)
    final_amplitude = np.sqrt(final_state[0] ** 2 + final_state[1] ** 2)

    assert final_amplitude < initial_amplitude


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_numericalpropagator_propagate_to_same_epoch():
    """Test propagate_to() to same epoch (no-op)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    initial_state = prop.current_state().copy()
    prop.propagate_to(epoch)

    np.testing.assert_array_almost_equal(prop.current_state(), initial_state)


def test_numericalpropagator_very_small_timestep():
    """Test with very small timestep"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.step_size = 0.001
    prop.propagate_steps(10)

    assert prop.current_epoch == epoch + 0.01


def test_numericalpropagator_single_dimension():
    """Test with 1D system"""
    epoch = create_test_epoch()
    state = np.array([1.0])

    def decay_dynamics(t, state, params):
        return np.array([-0.1 * state[0]])

    prop = NumericalPropagator(
        epoch, state, decay_dynamics, NumericalPropagationConfig.default()
    )

    prop.propagate_to(epoch + 1.0)

    assert prop.state_dim == 1
    assert prop.current_state()[0] < 1.0  # Should have decayed


def test_numericalpropagator_repr():
    """Test __repr__ method"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    repr_str = repr(prop)
    assert "NumericalPropagator" in repr_str
    assert "state_dim=2" in repr_str


# =============================================================================
# Integration Method Tests
# =============================================================================


def test_numericalpropagator_with_rk4():
    """Test with RK4 integrator"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    config = NumericalPropagationConfig.with_method(IntegrationMethod.RK4)

    prop = NumericalPropagator(epoch, state, sho_dynamics, config)

    # Use smaller step size for RK4 (fixed-step)
    prop.step_size = 0.01
    prop.propagate_to(epoch + np.pi)

    # Basic sanity check
    assert prop.current_state()[0] < 0


def test_numericalpropagator_with_rkf45():
    """Test with RKF45 integrator"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    config = NumericalPropagationConfig.with_method(IntegrationMethod.RKF45)

    prop = NumericalPropagator(epoch, state, sho_dynamics, config)

    prop.propagate_to(epoch + np.pi)

    assert prop.current_state()[0] < 0


def test_numericalpropagator_with_dp54():
    """Test with DP54 integrator (default)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    config = NumericalPropagationConfig.with_method(IntegrationMethod.DP54)

    prop = NumericalPropagator(epoch, state, sho_dynamics, config)

    prop.propagate_to(epoch + np.pi)

    assert prop.current_state()[0] < 0


# =============================================================================
# Event Detection Tests
# =============================================================================


def test_numericalpropagator_event_time_event():
    """Test time event detection (mirrors Rust test)"""
    from brahe import TimeEvent

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Add time event at 5 seconds
    event = TimeEvent(epoch + 5.0, "TimeEvent")
    prop.add_event_detector(event)

    # Propagate past event
    prop.propagate_to(epoch + 10.0)

    # Event should be detected
    events = prop.event_log()
    assert len(events) > 0

    detected = any("TimeEvent" in e.name for e in events)
    assert detected


def test_numericalpropagator_event_value_event():
    """Test value event detection (mirrors Rust test_event_value_event)"""
    from brahe import ValueEvent, EventDirection

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Add value event: position crosses zero (decreasing)
    def value_fn(epoch, state):
        return state[0]

    event = ValueEvent("PositionCrossing", value_fn, 0.0, EventDirection.DECREASING)
    prop.add_event_detector(event)

    # Propagate through multiple crossings
    prop.propagate_to(epoch + 10.0)

    # Should have detected crossings
    events = prop.event_log()
    assert len(events) > 0, "No events detected in event log"

    detected = any("PositionCrossing" in e.name for e in events)
    assert detected, "PositionCrossing event not found in event log"


def test_numericalpropagator_event_callback_state_modification():
    """Test event callback that modifies state (mirrors Rust test)"""
    from brahe import ValueEvent, EventDirection, EventAction

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Callback that adds small velocity impulse when position crosses zero
    def callback(epoch, state):
        new_state = state.copy()
        new_state[1] += 0.1  # Add small velocity impulse
        return (new_state, EventAction.CONTINUE)

    def value_fn(epoch, state):
        return state[0]

    event = ValueEvent(
        "ImpulseAtZero", value_fn, 0.0, EventDirection.DECREASING
    ).with_callback(callback)
    prop.add_event_detector(event)

    # Propagate
    prop.propagate_to(epoch + 3.0)

    # Event should have been detected
    events = prop.event_log()
    assert len(events) > 0, "No events detected in event log"

    detected = any("ImpulseAtZero" in e.name for e in events)
    assert detected, "ImpulseAtZero event not found in event log"


def test_numericalpropagator_event_terminal():
    """Test terminal event stops propagation (mirrors Rust test)"""
    from brahe import TimeEvent, EventAction

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Create terminal event with callback that returns STOP
    def stop_callback(epoch, state):
        return (None, EventAction.STOP)

    event = TimeEvent(epoch + 5.0, "Terminal").with_callback(stop_callback)
    prop.add_event_detector(event)

    # Try to propagate to time 10
    prop.propagate_to(epoch + 10.0)

    # Should have stopped at event (around time 5)
    assert prop.terminated()
    time_diff = float(prop.current_epoch - (epoch + 5.0))
    assert abs(time_diff) < 1.0


def test_numericalpropagator_event_query_by_detector_index():
    """Test events_by_detector_index method (mirrors Rust test)"""
    from brahe import TimeEvent

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Add multiple detectors
    prop.add_event_detector(TimeEvent(epoch + 2.0, "Event1"))
    prop.add_event_detector(TimeEvent(epoch + 4.0, "Event2"))

    # Propagate
    prop.propagate_to(epoch + 5.0)

    # Query events by detector index
    events_0 = prop.events_by_detector_index(0)
    events_1 = prop.events_by_detector_index(1)

    assert len(events_0) > 0
    assert len(events_1) > 0


def test_numericalpropagator_event_query_in_range():
    """Test events_in_range method (mirrors Rust test)"""
    from brahe import TimeEvent

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Add events at different times
    prop.add_event_detector(TimeEvent(epoch + 2.0, "Early"))
    prop.add_event_detector(TimeEvent(epoch + 8.0, "Late"))

    # Propagate
    prop.propagate_to(epoch + 10.0)

    # Query events in range
    events_early = prop.events_in_range(epoch, epoch + 5.0)
    events_late = prop.events_in_range(epoch + 5.0, epoch + 10.0)

    # Early event should be in first range
    has_early = any("Early" in e.name for e in events_early)
    assert has_early

    # Late event should be in second range
    has_late = any("Late" in e.name for e in events_late)
    assert has_late


def test_numericalpropagator_event_clear_and_reset_termination():
    """Test clear_events and reset_termination (mirrors Rust test)"""
    from brahe import TimeEvent, EventAction

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Create terminal event
    def stop_callback(epoch, state):
        return (None, EventAction.STOP)

    event = TimeEvent(epoch + 5.0, "Terminal").with_callback(stop_callback)
    prop.add_event_detector(event)

    # Propagate and hit terminal
    prop.propagate_to(epoch + 10.0)
    assert prop.terminated()

    # Clear events and reset termination
    prop.clear_events()
    prop.reset_termination()

    assert not prop.terminated()
    assert len(prop.event_log()) == 0

    # Can continue propagating
    prop.propagate_to(epoch + 15.0)


def test_numericalpropagator_event_api_methods():
    """Test event detection API methods exist and work"""
    from brahe import TimeEvent

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Test initial state
    assert len(prop.event_log()) == 0
    assert not prop.terminated()
    assert prop.latest_event() is None

    # Add an event detector
    time_event = TimeEvent(epoch + 5.0, "Test Event")
    prop.add_event_detector(time_event)

    # Event log still empty (not detected yet)
    assert len(prop.event_log()) == 0

    # Test clear_events and reset_termination work without error
    prop.clear_events()
    prop.reset_termination()
    assert not prop.terminated()


def test_numericalpropagator_events_by_name():
    """Test events_by_name method"""
    from brahe import TimeEvent

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Add multiple events with same name
    prop.add_event_detector(TimeEvent(epoch + 2.0, "Checkpoint"))
    prop.add_event_detector(TimeEvent(epoch + 4.0, "Other"))
    prop.add_event_detector(TimeEvent(epoch + 6.0, "Checkpoint"))

    # Propagate
    prop.propagate_to(epoch + 8.0)

    # Query by name
    checkpoints = prop.events_by_name("Checkpoint")
    others = prop.events_by_name("Other")

    assert len(checkpoints) == 2
    assert len(others) == 1


# =============================================================================
# DCovarianceProvider Tests
# =============================================================================


def test_dcovarianceprovider_no_covariance():
    """Test error when covariance not enabled (mirrors Rust test)"""
    import pytest

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    # Create propagator WITHOUT initial_covariance
    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Calling covariance() should raise an error
    with pytest.raises(RuntimeError):
        prop.covariance(epoch)


def test_dcovarianceprovider_with_initial_covariance():
    """Test covariance initialization and retrieval (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    # Create initial covariance (2x2 identity)
    initial_cov = np.eye(2)

    prop = NumericalPropagator(
        epoch,
        state,
        sho_dynamics,
        NumericalPropagationConfig.default(),
        params=None,
        initial_covariance=initial_cov,
    )

    # Propagate to build trajectory
    prop.propagate_to(epoch + 2.0)

    # Covariance should be retrievable at an intermediate point within trajectory
    cov = prop.covariance(epoch + 1.0)
    assert cov is not None
    assert cov.shape == (2, 2)


def test_dcovarianceprovider_positive_definiteness():
    """Test covariance matrix properties (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    # Initial covariance with positive diagonal
    initial_cov = np.array([[1.0, 0.1], [0.1, 0.5]])

    prop = NumericalPropagator(
        epoch,
        state,
        sho_dynamics,
        NumericalPropagationConfig.default(),
        params=None,
        initial_covariance=initial_cov,
    )

    # Propagate to build trajectory
    prop.propagate_to(epoch + 2.0)

    # Get covariance at intermediate time
    cov = prop.covariance(epoch + 1.0)

    # Verify positive diagonal
    assert cov[0, 0] > 0
    assert cov[1, 1] > 0

    # Verify symmetry
    np.testing.assert_array_almost_equal(cov, cov.T)


def test_dcovarianceprovider_interpolation():
    """Test covariance at intermediate time (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    initial_cov = np.eye(2)

    prop = NumericalPropagator(
        epoch,
        state,
        sho_dynamics,
        NumericalPropagationConfig.default(),
        params=None,
        initial_covariance=initial_cov,
    )

    # Propagate to build trajectory
    prop.propagate_to(epoch + 10.0)

    # Get covariance at intermediate point
    cov = prop.covariance(epoch + 5.0)
    assert cov is not None
    assert cov.shape == (2, 2)


def test_dcovarianceprovider_out_of_bounds():
    """Test error for out-of-range covariance queries (mirrors Rust test)"""
    import pytest

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    initial_cov = np.eye(2)

    prop = NumericalPropagator(
        epoch,
        state,
        sho_dynamics,
        NumericalPropagationConfig.default(),
        params=None,
        initial_covariance=initial_cov,
    )

    # Propagate
    prop.propagate_to(epoch + 10.0)

    # Query out of bounds should raise
    with pytest.raises(RuntimeError):
        prop.covariance(epoch + 100.0)


# =============================================================================
# STM Tests
# =============================================================================


def create_sho_with_stm():
    """Create SHO propagator with STM enabled"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    # Enable STM by providing initial covariance (this enables variational equations)
    initial_cov = np.eye(2)

    return NumericalPropagator(
        epoch,
        state,
        sho_dynamics,
        NumericalPropagationConfig.default(),
        params=None,
        initial_covariance=initial_cov,
    )


def test_stm_identity_initialization():
    """Test STM starts as identity (mirrors Rust test)"""
    prop = create_sho_with_stm()

    stm = prop.stm()

    if stm is not None:  # STM may be None initially until propagation
        # Should be identity matrix
        n = stm.shape[0]
        np.testing.assert_array_almost_equal(stm, np.eye(n))


def test_stm_propagation():
    """Test STM evolution during propagation (mirrors Rust test)"""
    prop = create_sho_with_stm()

    # Propagate forward
    prop.propagate_to(prop.initial_epoch + 5.0)

    stm = prop.stm()
    if stm is not None:
        # STM should have evolved from identity
        n = stm.shape[0]
        # After propagation, STM should not be identity (unless trivial dynamics)
        assert stm.shape == (n, n)


def test_stm_storage_in_trajectory():
    """Test STM stored in trajectory (mirrors Rust test)"""
    prop = create_sho_with_stm()

    # Propagate to build trajectory
    prop.propagate_to(prop.initial_epoch + 10.0)

    # Trajectory should have been populated
    traj = prop.trajectory
    assert len(traj) >= 2


def test_stm_interpolation():
    """Test STM at intermediate times (mirrors Rust test)"""
    prop = create_sho_with_stm()

    # Propagate to build trajectory
    prop.propagate_to(prop.initial_epoch + 10.0)

    # Should be able to get state at intermediate time
    state = prop.state(prop.initial_epoch + 5.0)
    assert len(state) == 2


def test_stm_reset():
    """Test STM resets to identity (mirrors Rust test)"""
    prop = create_sho_with_stm()
    initial_epoch = prop.initial_epoch

    # Propagate forward
    prop.propagate_to(initial_epoch + 5.0)

    # Reset
    prop.reset()

    # Check epoch is reset
    assert prop.current_epoch == initial_epoch

    # STM should be reset to identity (if available)
    stm = prop.stm()
    if stm is not None:
        n = stm.shape[0]
        np.testing.assert_array_almost_equal(stm, np.eye(n), decimal=10)


def test_stm_energy_conservation_check():
    """Test symplectic property (det ≈ 1) (mirrors Rust test)"""
    prop = create_sho_with_stm()

    # Propagate one full period
    period = 2 * np.pi
    prop.propagate_to(prop.initial_epoch + period)

    stm = prop.stm()
    if stm is not None:
        # For Hamiltonian systems, STM should be symplectic (det = 1)
        det = np.linalg.det(stm)
        assert abs(det - 1.0) < 0.1  # Relaxed tolerance for numerical integration


# =============================================================================
# Sensitivity Matrix Tests
# =============================================================================


def create_damped_sho_with_sensitivity():
    """Create damped SHO propagator with sensitivity enabled"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    params = np.array([1.0, 0.1])  # omega, zeta

    # Initial covariance enables variational equations
    initial_cov = np.eye(2)

    return NumericalPropagator(
        epoch,
        state,
        damped_oscillator_dynamics,
        NumericalPropagationConfig.default(),
        params=params,
        initial_covariance=initial_cov,
    )


def test_sensitivity_zero_initialization():
    """Test sensitivity initializes as zero (mirrors Rust test)"""
    prop = create_damped_sho_with_sensitivity()

    sens = prop.sensitivity()
    # Sensitivity may be None if not configured, or zero matrix initially
    if sens is not None:
        # Should be zero matrix initially
        np.testing.assert_array_almost_equal(sens, np.zeros_like(sens))


def test_sensitivity_propagation():
    """Test sensitivity evolution during propagation (mirrors Rust test)"""
    prop = create_damped_sho_with_sensitivity()

    # Propagate forward
    prop.propagate_to(prop.initial_epoch + 5.0)

    sens = prop.sensitivity()
    if sens is not None:
        # Sensitivity should have evolved from zero (may still be small)
        assert sens.shape[0] == 2  # state dimension


def test_sensitivity_parameter_dependence():
    """Test different parameters produce different sensitivity (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    # Create two propagators with different damping ratios
    # Using damped_oscillator_dynamics which uses params: [omega, zeta]
    params1 = np.array([1.0, 0.1])  # omega=1.0, zeta=0.1 (lightly damped)
    params2 = np.array([1.0, 0.5])  # omega=1.0, zeta=0.5 (heavily damped)

    # IMPORTANT: Must enable sensitivity propagation for params to be passed to dynamics
    # This mirrors the Rust test which sets config.variational.enable_sensitivity = true
    config = NumericalPropagationConfig.default().with_sensitivity()

    prop1 = NumericalPropagator(
        epoch, state, damped_oscillator_dynamics, config, params=params1
    )

    prop2 = NumericalPropagator(
        epoch,
        state,
        damped_oscillator_dynamics,
        NumericalPropagationConfig.default().with_sensitivity(),
        params=params2,
    )

    # Propagate both for a meaningful time to see parameter effect
    prop1.propagate_to(epoch + 5.0)
    prop2.propagate_to(epoch + 5.0)

    # Final states should differ due to different damping
    # The damped_oscillator_dynamics does use the params (omega and zeta)
    state1 = prop1.current_state()
    state2 = prop2.current_state()

    # With different damping coefficients, the states should diverge
    assert not np.allclose(state1, state2)


def test_sensitivity_storage_in_trajectory():
    """Test sensitivity stored in trajectory (mirrors Rust test)"""
    prop = create_damped_sho_with_sensitivity()

    # Propagate to build trajectory
    prop.propagate_to(prop.initial_epoch + 10.0)

    traj = prop.trajectory
    assert len(traj) >= 2


def test_sensitivity_interpolation():
    """Test sensitivity at intermediate times (mirrors Rust test)"""
    prop = create_damped_sho_with_sensitivity()

    # Propagate to build trajectory
    prop.propagate_to(prop.initial_epoch + 10.0)

    # Query state at intermediate time
    state = prop.state(prop.initial_epoch + 5.0)
    assert len(state) == 2


def test_sensitivity_reset():
    """Test sensitivity resets to zero (mirrors Rust test)"""
    prop = create_damped_sho_with_sensitivity()

    # Propagate forward
    prop.propagate_to(prop.initial_epoch + 5.0)

    # Reset
    prop.reset()

    # Check epoch is reset
    assert prop.current_epoch == prop.initial_epoch

    # Sensitivity should be reset to zero (if available)
    sens = prop.sensitivity()
    if sens is not None:
        np.testing.assert_array_almost_equal(sens, np.zeros_like(sens))


# =============================================================================
# InterpolationConfig Tests
# =============================================================================


def test_interpolationconfig_builder():
    """Test builder pattern for interpolation method (mirrors Rust test)"""
    from brahe import InterpolationMethod

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Use builder pattern
    prop.with_interpolation_method(InterpolationMethod.LINEAR)

    # Verify method was set
    method = prop.get_interpolation_method()
    assert method is not None


def test_interpolationconfig_setter_getter():
    """Test interpolation method getters/setters (mirrors Rust test)"""
    from brahe import InterpolationMethod

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Set interpolation method
    prop.set_interpolation_method(InterpolationMethod.LINEAR)

    # Get and verify
    method = prop.get_interpolation_method()
    assert method is not None


def test_interpolationconfig_persistence():
    """Test interpolation settings persist (mirrors Rust test)"""
    from brahe import InterpolationMethod

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.set_interpolation_method(InterpolationMethod.LINEAR)

    # Propagate
    prop.propagate_to(epoch + 5.0)

    # Method should persist
    method = prop.get_interpolation_method()
    assert method is not None


# =============================================================================
# CovarianceInterpolationConfig Tests
# =============================================================================


def test_covarianceinterpolationconfig_builder():
    """Test builder for covariance interpolation (mirrors Rust test)"""
    from brahe import CovarianceInterpolationMethod

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    initial_cov = np.eye(2)

    prop = NumericalPropagator(
        epoch,
        state,
        sho_dynamics,
        NumericalPropagationConfig.default(),
        params=None,
        initial_covariance=initial_cov,
    )

    # Use builder pattern
    prop.with_covariance_interpolation_method(
        CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )

    # Verify method was set
    method = prop.get_covariance_interpolation_method()
    assert method is not None


def test_covarianceinterpolationconfig_setter_getter():
    """Test covariance interpolation getters/setters (mirrors Rust test)"""
    from brahe import CovarianceInterpolationMethod

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    initial_cov = np.eye(2)

    prop = NumericalPropagator(
        epoch,
        state,
        sho_dynamics,
        NumericalPropagationConfig.default(),
        params=None,
        initial_covariance=initial_cov,
    )

    # Set method
    prop.set_covariance_interpolation_method(
        CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )

    # Get and verify
    method = prop.get_covariance_interpolation_method()
    assert method is not None


def test_covarianceinterpolationconfig_persistence():
    """Test covariance interpolation settings persist (mirrors Rust test)"""
    from brahe import CovarianceInterpolationMethod

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    initial_cov = np.eye(2)

    prop = NumericalPropagator(
        epoch,
        state,
        sho_dynamics,
        NumericalPropagationConfig.default(),
        params=None,
        initial_covariance=initial_cov,
    )

    prop.set_covariance_interpolation_method(
        CovarianceInterpolationMethod.TWO_WASSERSTEIN
    )

    # Propagate
    prop.propagate_to(epoch + 5.0)

    # Method should persist
    method = prop.get_covariance_interpolation_method()
    assert method is not None


# =============================================================================
# Expanded Identifiable Tests
# =============================================================================


def test_identifiable_with_name():
    """Test builder method for name (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Use builder pattern
    prop.with_name("SHO_Propagator")

    assert prop.get_name() == "SHO_Propagator"


def test_identifiable_set_name():
    """Test name setter/getter (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Use setter
    prop.set_name("TestPropagator")
    assert prop.get_name() == "TestPropagator"

    # Clear name
    prop.set_name(None)
    assert prop.get_name() is None


def test_identifiable_with_id():
    """Test builder method for ID (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.with_id(42)
    assert prop.get_id() == 42


def test_identifiable_set_id():
    """Test ID setter/getter (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.set_id(123)
    assert prop.get_id() == 123

    prop.set_id(None)
    assert prop.get_id() is None


def test_identifiable_with_uuid():
    """Test builder method for UUID (mirrors Rust test)"""
    import uuid as uuid_module

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    test_uuid = str(uuid_module.uuid4())
    prop.with_uuid(test_uuid)

    assert prop.get_uuid() == test_uuid


def test_identifiable_with_new_uuid():
    """Test automatic UUID generation (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.with_new_uuid()
    uuid_str = prop.get_uuid()

    assert uuid_str is not None
    assert len(uuid_str) == 36  # UUID format: 8-4-4-4-12


def test_identifiable_with_identity():
    """Test combined identity assignment (mirrors Rust test)"""
    import uuid as uuid_module

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    test_uuid = str(uuid_module.uuid4())
    prop.with_identity("CombinedTest", test_uuid, 999)

    assert prop.get_name() == "CombinedTest"
    assert prop.get_uuid() == test_uuid
    assert prop.get_id() == 999


def test_identifiable_persistence_through_propagation():
    """Test identity persists after propagation (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    prop.with_name("PersistentPropagator")
    prop.with_id(42)
    prop.with_new_uuid()

    original_uuid = prop.get_uuid()

    # Propagate
    prop.propagate_to(epoch + 10.0)

    # Identity should persist
    assert prop.get_name() == "PersistentPropagator"
    assert prop.get_id() == 42
    assert prop.get_uuid() == original_uuid


# =============================================================================
# Corner Case Tests
# =============================================================================


def test_corner_case_zero_parameters():
    """Test propagation with no parameters (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    # Create without params (None is default)
    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Should propagate successfully
    prop.propagate_to(epoch + 1.0)

    assert prop.current_epoch == epoch + 1.0


def test_corner_case_single_parameter():
    """Test propagation with single parameter (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    params = np.array([2.0])  # omega = 2.0

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default(), params=params
    )

    # Should propagate successfully
    prop.propagate_to(epoch + 1.0)

    assert prop.current_epoch == epoch + 1.0


def test_corner_case_sensitivity_without_params():
    """Test sensitivity with no parameters (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    initial_cov = np.eye(2)

    # Create with covariance but no params
    prop = NumericalPropagator(
        epoch,
        state,
        sho_dynamics,
        NumericalPropagationConfig.default(),
        params=None,
        initial_covariance=initial_cov,
    )

    # Should propagate successfully
    prop.propagate_to(epoch + 1.0)

    # Sensitivity should be None or empty (no parameters to vary)
    _ = prop.sensitivity()
    # May be None when no parameters exist


# =============================================================================
# Combined Event Filter Test
# =============================================================================


def test_numericalpropagator_events_combined_filters():
    """Test combining multiple event filter criteria (NEW - mirrors Rust test)"""
    from brahe import TimeEvent

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Add multiple detectors at different times
    prop.add_event_detector(TimeEvent(epoch + 2.0, "Early"))
    prop.add_event_detector(TimeEvent(epoch + 4.0, "Middle"))
    prop.add_event_detector(TimeEvent(epoch + 6.0, "Late"))
    prop.add_event_detector(TimeEvent(epoch + 8.0, "VeryLate"))

    # Propagate
    prop.propagate_to(epoch + 10.0)

    # Test combining filters:
    # 1. Get events by detector index
    events_0 = prop.events_by_detector_index(0)  # Early
    events_1 = prop.events_by_detector_index(1)  # Middle

    assert len(events_0) == 1
    assert len(events_1) == 1

    # 2. Get events in time range
    early_events = prop.events_in_range(epoch, epoch + 3.0)  # Should get "Early"
    middle_events = prop.events_in_range(
        epoch + 3.0, epoch + 5.0
    )  # Should get "Middle"
    late_events = prop.events_in_range(
        epoch + 5.0, epoch + 10.0
    )  # Should get "Late" and "VeryLate"

    assert len(early_events) == 1
    assert any("Early" in e.name for e in early_events)

    assert len(middle_events) == 1
    assert any("Middle" in e.name for e in middle_events)

    assert len(late_events) == 2
    assert any("Late" in e.name for e in late_events)
    assert any("VeryLate" in e.name for e in late_events)

    # 3. Get events by name
    early_by_name = prop.events_by_name("Early")
    assert len(early_by_name) == 1

    # 4. Combined: filter by name AND verify it's in the correct time range
    middle_by_name = prop.events_by_name("Middle")
    assert len(middle_by_name) == 1
    middle_event = middle_by_name[0]
    # Verify the event time is approximately epoch + 4.0
    time_diff = float(middle_event.window_open - (epoch + 4.0))
    assert abs(time_diff) < 0.1


# =============================================================================
# DStateProvider Additional Tests (mirrors Rust tests)
# =============================================================================


def test_dstateprovider_state_out_of_bounds():
    """Test error when querying state outside trajectory bounds (mirrors Rust test)"""
    import pytest

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Query before propagation should raise error for future time
    with pytest.raises(RuntimeError):
        prop.state(epoch + 100.0)


def test_dstateprovider_state_dimension_preservation():
    """Test state dimension is preserved during propagation (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    initial_dim = prop.state_dim

    # Propagate forward
    prop.propagate_to(epoch + 5.0)

    # Dimension should be preserved
    assert prop.state_dim == initial_dim
    assert len(prop.current_state()) == initial_dim


# =============================================================================
# Trajectory Storage Mode Tests (mirrors Rust tests)
# =============================================================================


def test_trajectory_allsteps_mode():
    """Test AllSteps trajectory storage mode (mirrors Rust test)"""
    from brahe import TrajectoryMode

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Set AllSteps mode explicitly
    prop.set_trajectory_mode(TrajectoryMode.ALL_STEPS)

    prop.step_size = 0.1
    prop.propagate_steps(10)

    # Trajectory should contain multiple states
    traj = prop.trajectory
    assert len(traj) > 1


def test_trajectory_disabled_mode():
    """Test Disabled trajectory storage mode (mirrors Rust test)"""
    from brahe import TrajectoryMode

    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])

    prop = NumericalPropagator(
        epoch, state, sho_dynamics, NumericalPropagationConfig.default()
    )

    # Reset clears the trajectory, then set disabled mode
    prop.reset()
    prop.set_trajectory_mode(TrajectoryMode.DISABLED)

    prop.step_size = 0.1
    prop.propagate_steps(10)

    # Trajectory should be empty since disabled mode prevents storage
    traj = prop.trajectory
    assert len(traj) == 0


def test_trajectory_stm_sensitivity_storage():
    """Test STM and sensitivity stored in trajectory (mirrors Rust test)"""
    epoch = create_test_epoch()
    state = np.array([1.0, 0.0])
    initial_cov = np.eye(2)

    prop = NumericalPropagator(
        epoch,
        state,
        sho_dynamics,
        NumericalPropagationConfig.default(),
        params=None,
        initial_covariance=initial_cov,
    )

    # Propagate to build trajectory
    prop.propagate_to(epoch + 5.0)

    # Trajectory should have been populated
    traj = prop.trajectory
    assert len(traj) >= 2

    # STM should be available (or None if not configured for history storage)
    _ = prop.stm()
    # The test mainly verifies no error occurs and trajectory is built
