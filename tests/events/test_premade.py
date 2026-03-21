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
    event = bh.EccentricityEvent(0.1, "Ecc value", bh.EventDirection.INCREASING)
    assert "Ecc" in str(event)


def test_InclinationEvent_new():
    """Test InclinationEvent constructor with degrees."""
    event = bh.InclinationEvent(
        45.0, "Inc value", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
    )
    assert "Inc" in str(event)


def test_InclinationEvent_radians():
    """Test InclinationEvent constructor with radians."""
    inc_rad = np.radians(45.0)
    event = bh.InclinationEvent(
        inc_rad, "Inc value", bh.EventDirection.ANY, bh.AngleFormat.RADIANS
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
    event = bh.SpeedEvent(7500.0, "Speed value", bh.EventDirection.INCREASING)
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


# =============================================================================
# AOI (Area of Interest) Events
# =============================================================================


def test_AOIEntryEvent_from_coordinates():
    """Test AOIEntryEvent construction from coordinate pairs."""
    vertices = [
        (10.0, 50.0),
        (20.0, 50.0),
        (20.0, 55.0),
        (10.0, 55.0),
        (10.0, 50.0),  # closed polygon
    ]
    event = bh.AOIEntryEvent.from_coordinates(
        vertices, "Europe AOI Entry", bh.AngleFormat.DEGREES
    )
    assert event is not None


def test_AOIEntryEvent_from_polygon():
    """Test AOIEntryEvent construction from PolygonLocation."""
    vertices = [
        np.array([10.0, 50.0, 0.0]),
        np.array([20.0, 50.0, 0.0]),
        np.array([20.0, 55.0, 0.0]),
        np.array([10.0, 55.0, 0.0]),
        np.array([10.0, 50.0, 0.0]),
    ]
    polygon = bh.PolygonLocation(vertices)
    event = bh.AOIEntryEvent(polygon, "Polygon AOI Entry")
    assert event is not None


def test_AOIEntryEvent_builder_chaining():
    """Test AOIEntryEvent builder pattern."""
    vertices = [
        (10.0, 50.0),
        (20.0, 50.0),
        (20.0, 55.0),
        (10.0, 55.0),
        (10.0, 50.0),
    ]
    event = (
        bh.AOIEntryEvent.from_coordinates(vertices, "AOI", bh.AngleFormat.DEGREES)
        .with_instance(2)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_AOIEntryEvent_radians():
    """Test AOIEntryEvent with radians input."""
    vertices_rad = [
        (np.radians(10.0), np.radians(50.0)),
        (np.radians(20.0), np.radians(50.0)),
        (np.radians(20.0), np.radians(55.0)),
        (np.radians(10.0), np.radians(55.0)),
        (np.radians(10.0), np.radians(50.0)),
    ]
    event = bh.AOIEntryEvent.from_coordinates(
        vertices_rad, "Radians AOI", bh.AngleFormat.RADIANS
    )
    assert event is not None


def test_AOIExitEvent_from_coordinates():
    """Test AOIExitEvent construction from coordinate pairs."""
    vertices = [
        (10.0, 50.0),
        (20.0, 50.0),
        (20.0, 55.0),
        (10.0, 55.0),
        (10.0, 50.0),
    ]
    event = bh.AOIExitEvent.from_coordinates(
        vertices, "Europe AOI Exit", bh.AngleFormat.DEGREES
    )
    assert event is not None


def test_AOIExitEvent_from_polygon():
    """Test AOIExitEvent construction from PolygonLocation."""
    vertices = [
        np.array([10.0, 50.0, 0.0]),
        np.array([20.0, 50.0, 0.0]),
        np.array([20.0, 55.0, 0.0]),
        np.array([10.0, 55.0, 0.0]),
        np.array([10.0, 50.0, 0.0]),
    ]
    polygon = bh.PolygonLocation(vertices)
    event = bh.AOIExitEvent(polygon, "Polygon AOI Exit")
    assert event is not None


def test_AOIExitEvent_builder_chaining():
    """Test AOIExitEvent builder pattern."""
    vertices = [
        (10.0, 50.0),
        (20.0, 50.0),
        (20.0, 55.0),
        (10.0, 55.0),
        (10.0, 50.0),
    ]
    event = (
        bh.AOIExitEvent.from_coordinates(vertices, "AOI", bh.AngleFormat.DEGREES)
        .with_instance(3)
        .with_tolerances(1e-4, 1e-7)
        .set_terminal()
    )
    assert event is not None


def test_AOI_antimeridian_polygon():
    """Test AOI events with polygon crossing the anti-meridian."""
    # Polygon in the Pacific crossing the anti-meridian
    vertices = [
        (170.0, -10.0),
        (-170.0, -10.0),  # crosses anti-meridian
        (-170.0, 10.0),
        (170.0, 10.0),
        (170.0, -10.0),
    ]
    entry_event = bh.AOIEntryEvent.from_coordinates(
        vertices, "Pacific Entry", bh.AngleFormat.DEGREES
    )
    exit_event = bh.AOIExitEvent.from_coordinates(
        vertices, "Pacific Exit", bh.AngleFormat.DEGREES
    )
    assert entry_event is not None
    assert exit_event is not None


# =============================================================================
# Builder Chaining Tests for Remaining Premade Events
# =============================================================================


def test_EccentricityEvent_builder_chaining():
    """Test EccentricityEvent builder pattern."""
    event = (
        bh.EccentricityEvent(0.1, "Ecc", bh.EventDirection.ANY)
        .with_instance(2)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_InclinationEvent_builder_chaining():
    """Test InclinationEvent builder pattern."""
    event = (
        bh.InclinationEvent(
            45.0, "Inc", bh.EventDirection.INCREASING, bh.AngleFormat.DEGREES
        )
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_ArgumentOfPerigeeEvent_builder_chaining():
    """Test ArgumentOfPerigeeEvent builder pattern."""
    event = (
        bh.ArgumentOfPerigeeEvent(
            90.0, "AoP", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
        )
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_MeanAnomalyEvent_builder_chaining():
    """Test MeanAnomalyEvent builder pattern."""
    event = (
        bh.MeanAnomalyEvent(
            0.0, "MA", bh.EventDirection.INCREASING, bh.AngleFormat.DEGREES
        )
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_EccentricAnomalyEvent_builder_chaining():
    """Test EccentricAnomalyEvent builder pattern."""
    event = (
        bh.EccentricAnomalyEvent(
            0.0, "EA", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
        )
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_TrueAnomalyEvent_builder_chaining():
    """Test TrueAnomalyEvent builder pattern."""
    event = (
        bh.TrueAnomalyEvent(180.0, "TA", bh.EventDirection.ANY, bh.AngleFormat.DEGREES)
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_ArgumentOfLatitudeEvent_builder_chaining():
    """Test ArgumentOfLatitudeEvent builder pattern."""
    event = (
        bh.ArgumentOfLatitudeEvent(
            30.0, "AoL", bh.EventDirection.INCREASING, bh.AngleFormat.DEGREES
        )
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_LatitudeEvent_builder_chaining():
    """Test LatitudeEvent builder pattern."""
    event = (
        bh.LatitudeEvent(
            0.0, "Equator", bh.EventDirection.INCREASING, bh.AngleFormat.DEGREES
        )
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_LongitudeEvent_builder_chaining():
    """Test LongitudeEvent builder pattern."""
    event = (
        bh.LongitudeEvent(
            0.0, "Greenwich", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
        )
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_PenumbraEvent_builder_chaining():
    """Test PenumbraEvent builder pattern."""
    event = (
        bh.PenumbraEvent("Penumbra", bh.EdgeType.RISING_EDGE, None)
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_EclipseEvent_builder_chaining():
    """Test EclipseEvent builder pattern."""
    event = (
        bh.EclipseEvent("Eclipse", bh.EdgeType.ANY_EDGE, None)
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_SunlitEvent_builder_chaining():
    """Test SunlitEvent builder pattern."""
    event = (
        bh.SunlitEvent("Sunlit", bh.EdgeType.RISING_EDGE, None)
        .with_instance(1)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


def test_DescendingNodeEvent_builder_chaining():
    """Test DescendingNodeEvent builder pattern."""
    event = (
        bh.DescendingNodeEvent("Desc Node")
        .with_instance(2)
        .with_tolerances(1e-5, 1e-8)
        .set_terminal()
    )
    assert event is not None


# =============================================================================
# str/repr tests for premade events
# =============================================================================


def test_premade_event_str_repr():
    """Test that str and repr work for various premade events."""
    events = [
        bh.SemiMajorAxisEvent(7000e3, "SMA Test", bh.EventDirection.ANY),
        bh.EccentricityEvent(0.01, "Ecc Test", bh.EventDirection.INCREASING),
        bh.AscendingNodeEvent("AN Test"),
        bh.DescendingNodeEvent("DN Test"),
        bh.AltitudeEvent(500e3, "Alt Test", bh.EventDirection.ANY),
        bh.SpeedEvent(7500.0, "Speed Test", bh.EventDirection.DECREASING),
        bh.UmbraEvent("Umbra Test", bh.EdgeType.ANY_EDGE, None),
        bh.EclipseEvent("Eclipse Test", bh.EdgeType.RISING_EDGE, None),
    ]
    for event in events:
        s = str(event)
        assert len(s) > 0
        r = repr(event)
        assert len(r) > 0
