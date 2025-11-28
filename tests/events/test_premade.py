"""Tests for premade event detectors."""

import brahe as bh
import numpy as np


# =============================================================================
# Orbital Element Events
# =============================================================================


def test_SemiMajorAxisEvent_new():
    """Test SemiMajorAxisEvent constructor."""
    event = bh.SemiMajorAxisEvent(7000e3, "SMA Check", bh.EventDirection.INCREASING)
    # Just verify construction works
    assert event is not None


def test_SemiMajorAxisEvent_builder_chaining():
    """Test SemiMajorAxisEvent builder pattern."""
    event = (
        bh.SemiMajorAxisEvent(7000e3, "SMA", bh.EventDirection.ANY)
        .with_instance(2)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    # If we get here without error, the chaining works
    assert event is not None


def test_EccentricityEvent_new():
    """Test EccentricityEvent constructor."""
    event = bh.EccentricityEvent(0.1, "Ecc Threshold", bh.EventDirection.INCREASING)
    assert "Ecc" in str(event)


def test_InclinationEvent_new():
    """Test InclinationEvent constructor with degrees."""
    event = bh.InclinationEvent(
        45.0, "Inc Threshold", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
    )
    assert "Inc" in str(event)


def test_InclinationEvent_radians():
    """Test InclinationEvent constructor with radians."""
    inc_rad = np.radians(45.0)
    event = bh.InclinationEvent(
        inc_rad, "Inc Threshold", bh.EventDirection.ANY, bh.AngleFormat.RADIANS
    )
    assert "Inc" in str(event)


def test_ArgumentOfPerigeeEvent_new():
    """Test ArgumentOfPerigeeEvent constructor with degrees."""
    event = bh.ArgumentOfPerigeeEvent(
        90.0, "AoP Check", bh.EventDirection.INCREASING, bh.AngleFormat.DEGREES
    )
    assert event is not None


def test_MeanAnomalyEvent_new():
    """Test MeanAnomalyEvent constructor with degrees."""
    event = bh.MeanAnomalyEvent(
        0.0, "Periapsis", bh.EventDirection.INCREASING, bh.AngleFormat.DEGREES
    )
    assert event is not None


def test_EccentricAnomalyEvent_new():
    """Test EccentricAnomalyEvent constructor with degrees."""
    event = bh.EccentricAnomalyEvent(
        0.0, "Periapsis EA", bh.EventDirection.INCREASING, bh.AngleFormat.DEGREES
    )
    assert event is not None


def test_TrueAnomalyEvent_new():
    """Test TrueAnomalyEvent constructor with degrees."""
    event = bh.TrueAnomalyEvent(
        180.0, "Apoapsis", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
    )
    assert event is not None


def test_ArgumentOfLatitudeEvent_new():
    """Test ArgumentOfLatitudeEvent constructor with degrees."""
    event = bh.ArgumentOfLatitudeEvent(
        30.0, "AoL Check", bh.EventDirection.INCREASING, bh.AngleFormat.DEGREES
    )
    assert event is not None


# =============================================================================
# Node Crossing Events
# =============================================================================


def test_AscendingNodeEvent_new():
    """Test AscendingNodeEvent constructor."""
    event = bh.AscendingNodeEvent("Ascending Node")
    assert event is not None


def test_AscendingNodeEvent_builder_chaining():
    """Test AscendingNodeEvent builder pattern."""
    event = (
        bh.AscendingNodeEvent("Asc Node").with_instance(2).with_tolerances(1e-5, 1e-8)
    )
    assert event is not None


def test_DescendingNodeEvent_new():
    """Test DescendingNodeEvent constructor."""
    event = bh.DescendingNodeEvent("Descending Node")
    assert event is not None


# =============================================================================
# State-Derived Events
# =============================================================================


def test_SpeedEvent_new():
    """Test SpeedEvent constructor."""
    event = bh.SpeedEvent(7500.0, "Speed Threshold", bh.EventDirection.INCREASING)
    assert event is not None


def test_SpeedEvent_builder_chaining():
    """Test SpeedEvent builder pattern."""
    event = (
        bh.SpeedEvent(7000.0, "Speed", bh.EventDirection.ANY)
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_LongitudeEvent_new():
    """Test LongitudeEvent constructor with degrees."""
    event = bh.LongitudeEvent(
        0.0, "Prime Meridian", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
    )
    assert event is not None


def test_LatitudeEvent_new():
    """Test LatitudeEvent constructor with degrees."""
    event = bh.LatitudeEvent(
        0.0, "Equator", bh.EventDirection.INCREASING, bh.AngleFormat.DEGREES
    )
    assert event is not None


# =============================================================================
# Eclipse/Shadow Events
# =============================================================================


def test_UmbraEvent_new_without_source():
    """Test UmbraEvent constructor without ephemeris source."""
    event = bh.UmbraEvent("Enter Umbra", bh.EdgeType.RISING_EDGE, None)
    assert event is not None


def test_UmbraEvent_new_with_source():
    """Test UmbraEvent constructor with ephemeris source."""
    event = bh.UmbraEvent(
        "Umbra DE440s", bh.EdgeType.FALLING_EDGE, bh.EphemerisSource.DE440s
    )
    assert event is not None


def test_UmbraEvent_builder_chaining():
    """Test UmbraEvent builder pattern."""
    event = (
        bh.UmbraEvent("Umbra", bh.EdgeType.RISING_EDGE, None)
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_PenumbraEvent_new():
    """Test PenumbraEvent constructor."""
    event = bh.PenumbraEvent("Penumbra", bh.EdgeType.RISING_EDGE, None)
    assert event is not None


def test_PenumbraEvent_with_source():
    """Test PenumbraEvent with ephemeris source."""
    event = bh.PenumbraEvent(
        "Penumbra DE440", bh.EdgeType.ANY_EDGE, bh.EphemerisSource.DE440
    )
    assert event is not None


def test_EclipseEvent_new():
    """Test EclipseEvent constructor."""
    event = bh.EclipseEvent("Eclipse", bh.EdgeType.ANY_EDGE, None)
    assert event is not None


def test_EclipseEvent_with_all_ephemeris_sources():
    """Test EclipseEvent with all ephemeris source variants."""
    # Low precision
    event1 = bh.EclipseEvent(
        "Eclipse LP", bh.EdgeType.RISING_EDGE, bh.EphemerisSource.LowPrecision
    )
    assert event1 is not None

    # DE440s
    event2 = bh.EclipseEvent(
        "Eclipse DE440s", bh.EdgeType.RISING_EDGE, bh.EphemerisSource.DE440s
    )
    assert event2 is not None

    # DE440
    event3 = bh.EclipseEvent(
        "Eclipse DE440", bh.EdgeType.RISING_EDGE, bh.EphemerisSource.DE440
    )
    assert event3 is not None


def test_SunlitEvent_new():
    """Test SunlitEvent constructor."""
    event = bh.SunlitEvent("Sunlit", bh.EdgeType.RISING_EDGE, None)
    assert event is not None


def test_SunlitEvent_with_source():
    """Test SunlitEvent with ephemeris source."""
    event = bh.SunlitEvent(
        "Sunlit DE440s", bh.EdgeType.FALLING_EDGE, bh.EphemerisSource.DE440s
    )
    assert event is not None


# =============================================================================
# Edge Type Tests
# =============================================================================


def test_all_edge_types():
    """Test that all edge types work with binary events."""
    for edge_type in [
        bh.EdgeType.RISING_EDGE,
        bh.EdgeType.FALLING_EDGE,
        bh.EdgeType.ANY_EDGE,
    ]:
        event = bh.EclipseEvent("Test", edge_type, None)
        assert event is not None


# =============================================================================
# Direction Tests
# =============================================================================


def test_all_event_directions():
    """Test that all event directions work with value events."""
    for direction in [
        bh.EventDirection.INCREASING,
        bh.EventDirection.DECREASING,
        bh.EventDirection.ANY,
    ]:
        event = bh.SemiMajorAxisEvent(7000e3, "Test", direction)
        assert event is not None
