"""Tests for common event detectors (TimeEvent, valueEvent, BinaryEvent)."""

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
