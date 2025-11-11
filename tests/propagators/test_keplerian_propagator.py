"""
Tests for KeplerianPropagator Python bindings

These tests mirror the Rust tests from src/orbits/keplerian_propagator.rs
"""

import numpy as np
import pytest
from brahe import (
    Epoch,
    TimeSystem,
    KeplerianPropagator,
    OrbitFrame,
    OrbitRepresentation,
    AngleFormat,
    orbital_period,
    state_osculating_to_cartesian,
    state_cartesian_to_osculating,
    state_eci_to_ecef,
    state_ecef_to_eci,
    state_itrf_to_gcrf,
    state_eme2000_to_gcrf,
    PanicException,
)

# Test data constants
TEST_EPOCH_JD = 2451545.0


def create_test_elements():
    """Create test Keplerian elements"""
    return np.array([7000e3, 0.01, 97.8, 15.0, 45.0, 60.0])


def create_circular_elements():
    """Create circular orbit Keplerian elements"""
    return np.array([7000e3, 0.0, 0.0, 0.0, 0.0, 0.0])


def create_cartesian_state():
    """Create test Cartesian state from test Keplerian elements"""
    a = 7000e3  # Semi-major axis in meters
    e = 0.01  # Eccentricity
    i = 97.8  # Inclination
    raan = 15.0  # Right Ascension of Ascending Node
    argp = 45.0  # Argument of perigee
    ma = 60.0  # Mean anomaly

    return state_osculating_to_cartesian(
        np.array([a, e, i, raan, argp, ma]), angle_format=AngleFormat.DEGREES
    )


# KeplerianPropagator Method Tests


def test_keplerianpropagator_new():
    """Test KeplerianPropagator.new()"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator(
        epoch,
        elements,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.RADIANS,
        60.0,
    )

    assert propagator.initial_epoch == epoch
    assert propagator.current_epoch == epoch
    state = propagator.initial_state()
    assert abs(state[0] - 7000e3) < 1.0
    assert abs(state[1] - 0.01) < 1e-10


def test_keplerianpropagator_new_invalid_angle_format():
    """Test that new() raises TypeError when angle_format is None for Keplerian elements"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    # This should raise TypeError because angle format cannot be None
    with pytest.raises(TypeError):
        KeplerianPropagator(
            epoch,
            elements,
            OrbitFrame.ECI,
            OrbitRepresentation.KEPLERIAN,
            None,  # Invalid: angle_format must be specified for Keplerian
            60.0,
        )


def test_keplerianpropagator_new_invalid_frame():
    """Test that new() panics with invalid frame"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    # This should panic because Keplerian elements are not in ECI frame
    with pytest.raises(PanicException):
        KeplerianPropagator(
            epoch,
            elements,
            OrbitFrame.ECEF,
            OrbitRepresentation.KEPLERIAN,
            AngleFormat.RADIANS,
            60.0,
        )


def test_keplerianpropagator_new_invalid_cartesian_angle_format():
    """Test that new() panics with invalid Cartesian angle format"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    state = create_test_elements()  # Using elements for simplicity

    # This should panic because angle format is not None for Cartesian
    with pytest.raises(PanicException):
        KeplerianPropagator(
            epoch,
            state,
            OrbitFrame.ECI,
            OrbitRepresentation.CARTESIAN,
            AngleFormat.RADIANS,
            60.0,
        )


def test_keplerianpropagator_new_invalid_step_size_negative():
    """Test that new() panics with negative step size"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    # This should panic because step size is not positive
    with pytest.raises(PanicException):
        KeplerianPropagator(
            epoch,
            elements,
            OrbitFrame.ECI,
            OrbitRepresentation.KEPLERIAN,
            AngleFormat.RADIANS,
            -10.0,
        )


def test_keplerianpropagator_new_invalid_step_size_zero():
    """Test that new() panics with zero step size"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    # This should panic because step size is not positive
    with pytest.raises(PanicException):
        KeplerianPropagator(
            epoch,
            elements,
            OrbitFrame.ECI,
            OrbitRepresentation.KEPLERIAN,
            AngleFormat.RADIANS,
            0.0,
        )


def test_keplerianpropagator_from_keplerian():
    """Test KeplerianPropagator.from_keplerian()"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    assert propagator.initial_epoch == epoch
    assert propagator.step_size == 60.0


def test_keplerianpropagator_from_eci():
    """Test KeplerianPropagator.from_eci()"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    cartesian = create_cartesian_state()

    propagator = KeplerianPropagator.from_eci(
        epoch,
        cartesian,
        60.0,
    )

    assert propagator.initial_epoch == epoch
    assert propagator.step_size == 60.0


def test_keplerianpropagator_from_ecef():
    """Test KeplerianPropagator.from_ecef()"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    cartesian = state_eci_to_ecef(epoch, create_cartesian_state())

    propagator = KeplerianPropagator.from_ecef(
        epoch,
        cartesian,
        60.0,
    )

    assert propagator.initial_epoch == epoch
    assert propagator.step_size == 60.0


# OrbitPropagator Trait Tests


def test_keplerianpropagator_orbitpropagator_step():
    """Test step() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    propagator.step()

    new_epoch = propagator.current_epoch
    assert new_epoch == epoch + 60.0
    assert len(propagator.trajectory) == 2

    # Confirm all elements except for mean anomaly are unchanged
    new_state = propagator.current_state()
    for i in range(5):
        assert abs(new_state[i] - elements[i]) < 1e-6
    # Mean anomaly should have changed
    assert new_state[5] != elements[5]


def test_keplerianpropagator_orbitpropagator_step_by():
    """Test step_by() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    propagator.step_by(120.0)

    new_epoch = propagator.current_epoch
    assert new_epoch == epoch + 120.0

    # Confirm only 2 states in trajectory (initial + 1 step)
    assert len(propagator.trajectory) == 2

    # Confirm all elements except for mean anomaly are unchanged
    new_state = propagator.current_state()
    for i in range(5):
        assert abs(new_state[i] - elements[i]) < 1e-6

    # Mean anomaly should have advanced by mean motion * 120s
    assert new_state[5] != elements[5]
    assert new_state[5] == elements[5] + (120.0 / orbital_period(elements[0])) * 360.0


def test_keplerian_orbitpropagator_step_past():
    """Test step_past() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    target_epoch = epoch + 250.0
    propagator.step_past(target_epoch)

    current_epoch = propagator.current_epoch
    assert current_epoch > target_epoch

    # Should have 6 steps: initial + 5 steps of 60s
    assert len(propagator.trajectory) == 6
    assert current_epoch == epoch + 300.0

    # Confirm all elements except for mean anomaly are unchanged
    new_state = propagator.current_state()
    for i in range(5):
        assert abs(new_state[i] - elements[i]) < 1e-6
    # Mean anomaly should have changed
    assert new_state[5] != elements[5]


def test_keplerianpropagator_orbitpropagator_propagate_steps():
    """Test propagate_steps() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    propagator.propagate_steps(5)

    assert len(propagator.trajectory) == 6  # Initial + 5 steps
    new_epoch = propagator.current_epoch
    assert new_epoch == epoch + 300.0

    # Confirm all elements except for mean anomaly are unchanged
    new_state = propagator.current_state()
    for i in range(5):
        assert abs(new_state[i] - elements[i]) < 1e-6
    # Mean anomaly should have changed
    assert new_state[5] != elements[5]


def test_keplerianpropagator_orbitpropagator_propagate_to():
    """Test propagate_to() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    target_epoch = epoch + 90.0
    propagator.propagate_to(target_epoch)

    current_epoch = propagator.current_epoch
    assert current_epoch == target_epoch

    # Should have 3 steps: initial + 1 step of 60s + 1 step of 30s
    assert len(propagator.trajectory) == 3

    # Confirm all elements except for mean anomaly are unchanged
    new_state = propagator.current_state()
    for i in range(5):
        assert abs(new_state[i] - elements[i]) < 1e-6
    # Mean anomaly should have changed
    assert new_state[5] != elements[5]


def test_keplerianpropagator_orbitpropagator_current_epoch():
    """Test current_epoch property"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    assert propagator.current_epoch == epoch

    # step and check epoch advanced
    propagator.step()
    assert propagator.current_epoch != epoch


def test_keplerianpropagator_orbitpropagator_current_state():
    """Test current_state() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    # Initial state should match
    np.testing.assert_array_equal(propagator.current_state(), elements)

    # After step, should be different
    propagator.step()
    current_state = propagator.current_state()
    assert not np.array_equal(current_state, elements)


def test_keplerianpropagator_orbitpropagator_initial_epoch():
    """Test initial_epoch property"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    assert propagator.initial_epoch == epoch

    # Step and confirm initial epoch unchanged
    propagator.step()
    assert propagator.initial_epoch == epoch


def test_keplerianpropagator_orbitpropagator_initial_state():
    """Test initial_state() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    np.testing.assert_array_equal(propagator.initial_state(), elements)

    # Step and confirm initial state unchanged
    propagator.step()
    np.testing.assert_array_equal(propagator.initial_state(), elements)


def test_keplerianpropagator_orbitpropagator_step_size():
    """Test step_size property"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    assert propagator.step_size == 60.0


def test_keplerianpropagator_orbitpropagator_set_step_size():
    """Test setting step_size"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    # Confirm initial step size
    assert propagator.step_size == 60.0

    # Change step size
    propagator.step_size = 120.0
    assert propagator.step_size == 120.0


def test_keplerianpropagator_set_step_size_method():
    """Test set_step_size explicit method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    # Test explicit method call (in addition to property setter)
    propagator.set_step_size(120.0)

    assert propagator.step_size == 120.0

    # Test that both property and method work interchangeably
    propagator.step_size = 90.0
    assert propagator.step_size == 90.0

    propagator.set_step_size(150.0)
    assert propagator.step_size == 150.0


def test_keplerianpropagator_orbitpropagator_reset():
    """Test reset() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    # Propagate forward
    propagator.propagate_steps(5)
    assert len(propagator.trajectory) == 6

    # Reset
    propagator.reset()
    assert len(propagator.trajectory) == 1
    assert propagator.current_epoch == epoch


def test_keplerianpropagator_orbitpropagator_set_initial_conditions():
    """Test set_initial_conditions() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    # Set new initial conditions
    new_epoch = Epoch.from_jd(TEST_EPOCH_JD + 1.0, TimeSystem.UTC)
    new_elements = create_circular_elements()

    propagator.set_initial_conditions(
        new_epoch,
        new_elements,
        OrbitFrame.ECI,
        OrbitRepresentation.KEPLERIAN,
        AngleFormat.RADIANS,
    )

    assert propagator.initial_epoch == new_epoch
    np.testing.assert_array_equal(propagator.initial_state(), new_elements)


def test_keplerianpropagator_orbitpropagator_set_eviction_policy_max_size():
    """Test set_eviction_policy_max_size() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    propagator.set_eviction_policy_max_size(5)

    # Propagate 10 steps
    propagator.propagate_steps(10)

    # Should only keep 5 states
    assert len(propagator.trajectory) == 5


def test_keplerianpropagator_orbitpropagator_set_eviction_policy_max_age():
    """Test set_eviction_policy_max_age() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    # Set eviction policy - only keep states within 120 seconds of current
    propagator.set_eviction_policy_max_age(120.0)

    # Propagate several steps (10 * 60s = 600s total)
    propagator.propagate_steps(10)

    # Should have evicted old states - should keep only last ~3 states (120s / 60s step)
    # Plus current state: 3 previous + current = 4 states max
    assert len(propagator.trajectory) <= 4
    assert len(propagator.trajectory) > 0


# StateProvider Trait Tests


def test_keplerianpropagator_analyticpropagator_state():
    """Test state() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    target_epoch = epoch + orbital_period(elements[0])
    state = propagator.state(target_epoch)

    # State should be exactly the same as initial elements after one orbital period
    for i in range(6):
        assert abs(state[i] - elements[i]) < 1e-9


def test_keplerianpropagator_analyticpropagator_state_eci():
    """Test state_eci() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    state = propagator.state_eci(epoch + orbital_period(elements[0]))

    # Should be Cartesian state in ECI
    assert np.linalg.norm(state) > 0.0
    # Convert back to orbital elements and verify semi-major axis is preserved
    computed_elements = state_cartesian_to_osculating(
        state, angle_format=AngleFormat.DEGREES
    )

    # Confirm equality within small tolerance
    for i in range(6):
        assert abs(computed_elements[i] - elements[i]) < 1e-6


def test_keplerianpropagator_analyticpropagator_state_ecef():
    """Test state_ecef() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    state = propagator.state_ecef(epoch + orbital_period(elements[0]))

    # Convert back into osculating elements via ECI
    eci_state = state_ecef_to_eci(epoch + orbital_period(elements[0]), state)
    computed_elements = state_cartesian_to_osculating(
        eci_state, angle_format=AngleFormat.DEGREES
    )

    # Confirm equality within small tolerance
    for i in range(6):
        assert abs(computed_elements[i] - elements[i]) < 1e-6


def test_keplerianpropagator_analyticpropagator_state_gcrf():
    """Test state_gcrf() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    state = propagator.state_gcrf(epoch + orbital_period(elements[0]))

    # Convert back into osculating elements (GCRF is inertial, direct conversion)
    computed_elements = state_cartesian_to_osculating(
        state, angle_format=AngleFormat.DEGREES
    )

    # Confirm equality within small tolerance
    for i in range(6):
        assert abs(computed_elements[i] - elements[i]) < 1e-6


def test_keplerianpropagator_analyticpropagator_state_itrf():
    """Test state_itrf() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    state = propagator.state_itrf(epoch + orbital_period(elements[0]))

    # Convert back into osculating elements via GCRF
    gcrf_state = state_itrf_to_gcrf(epoch + orbital_period(elements[0]), state)
    computed_elements = state_cartesian_to_osculating(
        gcrf_state, angle_format=AngleFormat.DEGREES
    )

    # Confirm equality within small tolerance
    for i in range(6):
        assert abs(computed_elements[i] - elements[i]) < 1e-6


def test_keplerianpropagator_analyticpropagator_state_eme2000():
    """Test state_eme2000() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    state = propagator.state_eme2000(epoch + orbital_period(elements[0]))

    # Convert back into osculating elements via GCRF
    gcrf_state = state_eme2000_to_gcrf(state)
    computed_elements = state_cartesian_to_osculating(
        gcrf_state, angle_format=AngleFormat.DEGREES
    )

    # Confirm equality within small tolerance
    for i in range(6):
        assert abs(computed_elements[i] - elements[i]) < 1e-6


def test_keplerianpropagator_analyticpropagator_state_as_osculating_elements():
    """Test state_as_osculating_elements() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    osc_elements = propagator.state_as_osculating_elements(
        epoch + orbital_period(elements[0]), angle_format=AngleFormat.DEGREES
    )

    # Should match initial elements within small tolerance
    for i in range(6):
        assert abs(osc_elements[i] - elements[i]) < 1e-6

    # Now test with radians to degrees conversion
    osc_elements_rad = propagator.state_as_osculating_elements(
        epoch + orbital_period(elements[0]), angle_format=AngleFormat.RADIANS
    )
    for i in range(2):
        assert abs(osc_elements_rad[i] - elements[i]) < 1e-6
    for i in range(2, 6):
        assert abs(np.rad2deg(osc_elements_rad[i]) - elements[i]) < 1e-6


def test_keplerianpropagator_analyticpropagator_states():
    """Test states() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    epochs = [
        epoch,
        epoch + orbital_period(elements[0]),
        epoch + 2.0 * orbital_period(elements[0]),
    ]

    traj = propagator.states(epochs)
    assert len(traj) == 3

    # Confirm all elements remain unchanged within small tolerance
    for state in traj:
        for i in range(6):
            assert abs(state[i] - elements[i]) < 1e-6


def test_keplerianpropagator_analyticpropagator_states_eci():
    """Test states_eci() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    epochs = [
        epoch,
        epoch + orbital_period(elements[0]),
        epoch + 2.0 * orbital_period(elements[0]),
    ]

    states = propagator.states_eci(epochs)
    assert len(states) == 3
    # Verify states convert back to original elements within small tolerance
    for state in states:
        computed_elements = state_cartesian_to_osculating(
            state, angle_format=AngleFormat.DEGREES
        )
        for i in range(6):
            assert abs(computed_elements[i] - elements[i]) < 1e-6


def test_keplerianpropagator_analyticpropagator_states_ecef():
    """Test states_ecef() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    epochs = [
        epoch,
        epoch + orbital_period(elements[0]),
        epoch + 2.0 * orbital_period(elements[0]),
    ]

    states = propagator.states_ecef(epochs)
    assert len(states) == 3
    # Verify states convert back to original elements within small tolerance
    for i, state in enumerate(states):
        eci_state = state_ecef_to_eci(epochs[i], state)
        computed_elements = state_cartesian_to_osculating(
            eci_state, angle_format=AngleFormat.DEGREES
        )
        for j in range(6):
            assert abs(computed_elements[j] - elements[j]) < 1e-6


def test_keplerianpropagator_analyticpropagator_states_as_osculating_elements():
    """Test states_as_osculating_elements() method"""
    epoch = Epoch.from_jd(TEST_EPOCH_JD, TimeSystem.UTC)
    elements = create_test_elements()

    propagator = KeplerianPropagator.from_keplerian(
        epoch,
        elements,
        AngleFormat.DEGREES,
        60.0,
    )

    epochs = [
        epoch,
        epoch + orbital_period(elements[0]),
        epoch + 2.0 * orbital_period(elements[0]),
    ]

    traj = propagator.states_as_osculating_elements(
        epochs, angle_format=AngleFormat.DEGREES
    )
    assert len(traj) == 3

    # Confirm all elements remain unchanged within small tolerance
    for state in traj:
        for i in range(6):
            assert abs(state[i] - elements[i]) < 1e-6

    # Repeat with radians output
    traj_rad = propagator.states_as_osculating_elements(
        epochs, angle_format=AngleFormat.RADIANS
    )
    assert len(traj_rad) == 3

    for state in traj_rad:
        for i in range(2):
            assert abs(state[i] - elements[i]) < 1e-6
        for i in range(2, 6):
            assert abs(np.rad2deg(state[i]) - elements[i]) < 1e-6
