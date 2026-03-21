"""Tests for common event detectors (TimeEvent, ValueEvent, BinaryEvent)."""

import brahe as bh


def test_time_event_custom_tolerance():
    """Test TimeEvent with custom tolerance."""
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Test method chaining
    _ = bh.TimeEvent(epoch, "Custom").with_time_tolerance(1e-3)

    # Test chaining with other methods
    _ = (
        bh.TimeEvent(epoch, "Chained")
        .with_time_tolerance(5e-4)
        .with_instance(2)
        .set_terminal()
    )


def test_time_event_builder_pattern():
    """Test that TimeEvent builder methods can be chained in any order."""
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Order 1: tolerance -> instance -> terminal
    _ = (
        bh.TimeEvent(epoch, "Test")
        .with_time_tolerance(1e-4)
        .with_instance(1)
        .set_terminal()
    )

    # Order 2: instance -> tolerance -> terminal
    _ = (
        bh.TimeEvent(epoch, "Test")
        .with_instance(1)
        .with_time_tolerance(1e-4)
        .set_terminal()
    )

    # Order 3: terminal -> tolerance -> instance
    _ = (
        bh.TimeEvent(epoch, "Test")
        .set_terminal()
        .with_time_tolerance(1e-4)
        .with_instance(1)
    )


# =============================================================================
# ValueEvent tests
# =============================================================================


def _value_fn(epoch, state):
    """Sample value function for ValueEvent tests."""
    return state[0]


def _binary_fn(epoch, state):
    """Sample binary function for BinaryEvent tests."""
    return state[0] > 0.0


def test_value_event_basic():
    """Test ValueEvent constructor with all event directions."""
    for direction in [
        bh.EventDirection.INCREASING,
        bh.EventDirection.DECREASING,
        bh.EventDirection.ANY,
    ]:
        event = bh.ValueEvent("Test Value", _value_fn, 100.0, direction)
        assert event is not None


def test_value_event_builder_chain():
    """Test ValueEvent full builder chain."""
    event = (
        bh.ValueEvent("Full Chain", _value_fn, 100.0, bh.EventDirection.INCREASING)
        .with_tolerances(1e-5, 1e-8)
        .with_instance(3)
        .set_terminal()
    )
    assert event is not None


def test_value_event_builder_chain_reverse_order():
    """Test ValueEvent builder in different order."""
    event = (
        bh.ValueEvent("Reverse", _value_fn, 42.0, bh.EventDirection.DECREASING)
        .set_terminal()
        .with_instance(1)
        .with_tolerances(1e-4, 1e-7)
    )
    assert event is not None


def test_value_event_str_repr():
    """Test ValueEvent string representations."""
    event = bh.ValueEvent("My Value Event", _value_fn, 100.0, bh.EventDirection.ANY)
    r = repr(event)
    assert len(r) > 0


# =============================================================================
# BinaryEvent tests
# =============================================================================


def test_binary_event_basic():
    """Test BinaryEvent constructor with all edge types."""
    for edge_type in [
        bh.EdgeType.RISING_EDGE,
        bh.EdgeType.FALLING_EDGE,
        bh.EdgeType.ANY_EDGE,
    ]:
        event = bh.BinaryEvent("Test Binary", _binary_fn, edge_type)
        assert event is not None


def test_binary_event_builder_chain():
    """Test BinaryEvent full builder chain."""
    event = (
        bh.BinaryEvent("Full Chain", _binary_fn, bh.EdgeType.RISING_EDGE)
        .with_instance(2)
        .set_terminal()
    )
    assert event is not None


def test_binary_event_str_repr():
    """Test BinaryEvent string representations."""
    event = bh.BinaryEvent("My Binary Event", _binary_fn, bh.EdgeType.ANY_EDGE)
    r = repr(event)
    assert len(r) > 0


# =============================================================================
# TimeEvent str/repr tests
# =============================================================================


def test_time_event_str_repr():
    """Test TimeEvent string representations."""
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    event = bh.TimeEvent(epoch, "My Time Event")
    r = repr(event)
    assert len(r) > 0


# =============================================================================
# Enum tests
# =============================================================================


def test_event_direction_variants():
    """Test EventDirection enum values."""
    assert bh.EventDirection.INCREASING != bh.EventDirection.DECREASING
    assert bh.EventDirection.ANY != bh.EventDirection.INCREASING
    assert bh.EventDirection.INCREASING == bh.EventDirection.INCREASING


def test_edge_type_variants():
    """Test EdgeType enum values."""
    assert bh.EdgeType.RISING_EDGE != bh.EdgeType.FALLING_EDGE
    assert bh.EdgeType.ANY_EDGE != bh.EdgeType.RISING_EDGE
    assert bh.EdgeType.RISING_EDGE == bh.EdgeType.RISING_EDGE
