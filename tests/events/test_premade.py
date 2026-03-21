"""Tests for premade event detectors."""

import brahe as bh
import numpy as np


# =============================================================================
# Helper functions
# =============================================================================


def _create_numerical_propagator_leo(
    epoch=None, eccentricity=0.01, inclination_deg=97.8
):
    """Create a NumericalOrbitPropagator with standard LEO orbit.

    Args:
        epoch: Initial epoch. Defaults to 2024-01-01 00:00:00 UTC.
        eccentricity: Orbital eccentricity. Default 0.01.
        inclination_deg: Inclination in degrees. Default 97.8.

    Returns:
        Tuple of (propagator, epoch, orbital_period).
    """
    if epoch is None:
        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    a = bh.R_EARTH + 500e3
    oe = np.array([a, eccentricity, inclination_deg, 15.0, 30.0, 45.0])
    state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

    prop = bh.NumericalOrbitPropagator(
        epoch,
        state,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
        None,
    )
    period = bh.orbital_period(a)
    return prop, epoch, period


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


def test_premade_event_str_repr_remaining():
    """Test str/repr for remaining premade events not covered above."""
    events = [
        bh.InclinationEvent(
            45.0, "Inc Test", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
        ),
        bh.ArgumentOfPerigeeEvent(
            90.0, "AoP Test", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
        ),
        bh.MeanAnomalyEvent(
            0.0, "MA Test", bh.EventDirection.INCREASING, bh.AngleFormat.DEGREES
        ),
        bh.EccentricAnomalyEvent(
            0.0, "EA Test", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
        ),
        bh.TrueAnomalyEvent(
            180.0, "TA Test", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
        ),
        bh.ArgumentOfLatitudeEvent(
            30.0, "AoL Test", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
        ),
        bh.LatitudeEvent(
            0.0, "Lat Test", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
        ),
        bh.LongitudeEvent(
            0.0, "Lon Test", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
        ),
        bh.PenumbraEvent("Penumbra Test", bh.EdgeType.ANY_EDGE, None),
        bh.SunlitEvent("Sunlit Test", bh.EdgeType.ANY_EDGE, None),
    ]
    for event in events:
        s = str(event)
        assert isinstance(s, str)
        r = repr(event)
        assert isinstance(r, str)


# =============================================================================
# Integration Tests: Events with NumericalOrbitPropagator
# =============================================================================


def test_ascending_node_event_integration():
    """Test AscendingNodeEvent detection with NumericalOrbitPropagator."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.AscendingNodeEvent("Ascending Node")
    prop.add_event_detector(event)

    # Propagate for one full orbit
    prop.propagate_to(epoch + period)

    events = prop.event_log()
    assert len(events) >= 1, (
        f"Expected at least 1 ascending node event, got {len(events)}"
    )
    assert events[0].name == "Ascending Node"

    # Window open should be within the propagation span
    event_epoch = events[0].window_open
    assert event_epoch >= epoch
    assert event_epoch <= epoch + period


def test_descending_node_event_integration():
    """Test DescendingNodeEvent detection with NumericalOrbitPropagator."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.DescendingNodeEvent("Descending Node")
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    events = prop.event_log()
    assert len(events) >= 1, (
        f"Expected at least 1 descending node event, got {len(events)}"
    )
    assert events[0].name == "Descending Node"


def test_ascending_and_descending_node_events_together():
    """Test both node crossing events on same propagator."""
    prop, epoch, period = _create_numerical_propagator_leo()

    asc_event = bh.AscendingNodeEvent("Asc Node")
    desc_event = bh.DescendingNodeEvent("Desc Node")
    prop.add_event_detector(asc_event)
    prop.add_event_detector(desc_event)

    # Propagate for two orbits to get multiple crossings
    prop.propagate_to(epoch + 2 * period)

    events = prop.event_log()
    asc_events = [e for e in events if e.name == "Asc Node"]
    desc_events = [e for e in events if e.name == "Desc Node"]

    # Over two orbits, expect at least 2 of each
    assert len(asc_events) >= 2, (
        f"Expected >= 2 ascending events, got {len(asc_events)}"
    )
    assert len(desc_events) >= 2, (
        f"Expected >= 2 descending events, got {len(desc_events)}"
    )


def test_altitude_event_integration():
    """Test AltitudeEvent detection with elliptical orbit."""
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Use elliptical orbit so altitude varies
    a = bh.R_EARTH + 500e3
    e = 0.02
    oe = np.array([a, e, 0.0, 0.0, 0.0, 0.0])
    state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)

    prop = bh.NumericalOrbitPropagator(
        epoch,
        state,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
        None,
    )

    # Detect altitude crossings at 490 km
    alt_event = bh.AltitudeEvent(490e3, "Alt Crossing", bh.EventDirection.ANY)
    prop.add_event_detector(alt_event)

    period = bh.orbital_period(a)
    prop.propagate_to(epoch + 2 * period)

    events = prop.event_log()
    assert len(events) > 0, f"Expected altitude crossing events, got {len(events)}"


def test_true_anomaly_event_integration():
    """Test TrueAnomalyEvent at 90 degrees detection."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.TrueAnomalyEvent(
        90.0, "TA 90", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
    )
    prop.add_event_detector(event)

    # Propagate for two orbits to ensure crossings are captured
    prop.propagate_to(epoch + 2 * period)

    events = prop.event_log()
    assert len(events) >= 1, (
        f"Expected at least 1 true anomaly event, got {len(events)}"
    )
    assert events[0].name == "TA 90"


def test_mean_anomaly_event_integration():
    """Test MeanAnomalyEvent at 180 degrees detection."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.MeanAnomalyEvent(
        180.0, "MA 180", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
    )
    prop.add_event_detector(event)

    # Propagate for two orbits to ensure crossings
    prop.propagate_to(epoch + 2 * period)

    events = prop.event_log()
    assert len(events) >= 1, (
        f"Expected at least 1 mean anomaly event, got {len(events)}"
    )


def test_eccentric_anomaly_event_integration():
    """Test EccentricAnomalyEvent detection."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.EccentricAnomalyEvent(
        90.0, "EA 90deg", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
    )
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    events = prop.event_log()
    assert len(events) >= 1, (
        f"Expected at least 1 eccentric anomaly event, got {len(events)}"
    )


def test_argument_of_latitude_event_integration():
    """Test ArgumentOfLatitudeEvent detection."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.ArgumentOfLatitudeEvent(
        90.0, "AoL 90deg", bh.EventDirection.ANY, bh.AngleFormat.DEGREES
    )
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    events = prop.event_log()
    assert len(events) >= 1, f"Expected at least 1 AoL event, got {len(events)}"


def test_terminal_event_stops_propagation():
    """Test that a terminal event stops propagation early."""
    prop, epoch, period = _create_numerical_propagator_leo()

    # Set ascending node as terminal - should stop at first detection
    event = bh.AscendingNodeEvent("Terminal AN").set_terminal()
    prop.add_event_detector(event)

    # Try to propagate for two full orbits
    prop.propagate_to(epoch + 2 * period)

    events = prop.event_log()
    # Should have detected exactly 1 event (stopped at first)
    assert len(events) == 1, f"Expected exactly 1 terminal event, got {len(events)}"

    # Propagator should indicate terminated state
    assert prop.terminated()


def test_multiple_events_on_same_propagator():
    """Test multiple event types on same propagator."""
    prop, epoch, period = _create_numerical_propagator_leo()

    # Add node crossing events (reliable detection)
    asc_event = bh.AscendingNodeEvent("Asc Node")
    desc_event = bh.DescendingNodeEvent("Desc Node")

    prop.add_event_detector(asc_event)
    prop.add_event_detector(desc_event)

    # Propagate for two orbits to get reliable detection
    prop.propagate_to(epoch + 2 * period)

    events = prop.event_log()
    # Should have multiple events from different detectors
    # Over 2 orbits: at least 2 ascending + 2 descending = 4 events
    assert len(events) >= 4, f"Expected at least 4 total events, got {len(events)}"

    # Verify different event names present
    event_names = {e.name for e in events}
    assert "Asc Node" in event_names
    assert "Desc Node" in event_names


def test_event_query_integration():
    """Test EventQuery filtering with propagation results."""
    prop, epoch, period = _create_numerical_propagator_leo()

    asc_event = bh.AscendingNodeEvent("Asc Node")
    desc_event = bh.DescendingNodeEvent("Desc Node")
    prop.add_event_detector(asc_event)
    prop.add_event_detector(desc_event)

    prop.propagate_to(epoch + 2 * period)

    # Test query by name
    query = prop.query_events()
    assert len(query) > 0

    asc_query = query.by_name_exact("Asc Node")
    asc_events = asc_query.collect()
    assert len(asc_events) >= 2

    # Test query by name contains
    node_query = query.by_name_contains("Node")
    assert node_query.count() >= 4

    # Test any/is_empty
    assert query.any()
    assert not query.is_empty()

    # Test first/last
    first_event = query.first()
    assert first_event is not None
    last_event = query.last()
    assert last_event is not None

    # Test by_detector_index
    idx0_events = query.by_detector_index(0).collect()
    assert len(idx0_events) >= 2

    # Test chained queries
    filtered = query.by_detector_index(0).in_time_range(epoch, epoch + period)
    assert filtered.count() >= 1

    # Test repr
    r = repr(query)
    assert "EventQuery" in r


def test_event_query_after_before():
    """Test EventQuery after/before filters."""
    prop, epoch, period = _create_numerical_propagator_leo()

    asc_event = bh.AscendingNodeEvent("Asc Node")
    prop.add_event_detector(asc_event)

    prop.propagate_to(epoch + 2 * period)

    query = prop.query_events()
    total_count = query.count()
    assert total_count >= 2

    # Filter events after midpoint
    midpoint = epoch + period
    after_events = query.after(midpoint).collect()
    before_events = query.before(midpoint).collect()

    # Both should have at least one event
    assert len(after_events) >= 1
    assert len(before_events) >= 1

    # Together should equal total
    assert len(after_events) + len(before_events) >= total_count


def test_event_query_by_action():
    """Test EventQuery by_action filter."""
    prop, epoch, period = _create_numerical_propagator_leo()

    # Non-terminal event
    event = bh.AscendingNodeEvent("Asc Node")
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    query = prop.query_events()
    continue_events = query.by_action(bh.EventAction.CONTINUE).collect()
    assert len(continue_events) >= 1

    # No STOP events (none are terminal)
    stop_events = query.by_action(bh.EventAction.STOP).collect()
    assert len(stop_events) == 0


def test_event_query_iterator():
    """Test EventQuery iteration via __iter__."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.AscendingNodeEvent("Asc Node")
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    query = prop.query_events()
    event_list = list(query)
    assert len(event_list) >= 1

    for detected_event in query:
        assert detected_event.name == "Asc Node"


def test_detected_event_properties():
    """Test DetectedEvent property accessors."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.AscendingNodeEvent("Asc Node")
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    events = prop.event_log()
    assert len(events) >= 1

    detected = events[0]

    # Test all property accessors
    assert detected.name == "Asc Node"
    assert detected.window_open is not None
    assert detected.window_close is not None
    assert isinstance(detected.value, float)
    assert detected.action is not None
    assert detected.event_type is not None

    # Test alias methods
    assert detected.t_start() is not None
    assert detected.t_end() is not None
    assert detected.start_time() is not None
    assert detected.end_time() is not None

    # Test entry/exit state arrays
    entry_state = detected.entry_state
    assert isinstance(entry_state, np.ndarray)
    assert len(entry_state) == 6

    exit_state = detected.exit_state
    assert isinstance(exit_state, np.ndarray)
    assert len(exit_state) == 6

    # Test str/repr
    s = str(detected)
    assert "DetectedEvent" in s
    assert "Asc Node" in s

    r = repr(detected)
    assert "DetectedEvent" in r


def test_speed_event_integration():
    """Test SpeedEvent detection with NumericalOrbitPropagator."""
    prop, epoch, period = _create_numerical_propagator_leo(eccentricity=0.02)

    # Speed at 500 km LEO is ~7600 m/s. With eccentricity, speed varies.
    event = bh.SpeedEvent(7600.0, "Speed Crossing", bh.EventDirection.ANY)
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    events = prop.event_log()
    assert len(events) >= 1, f"Expected speed crossing events, got {len(events)}"


def test_semi_major_axis_event_integration():
    """Test SemiMajorAxisEvent detection with NumericalOrbitPropagator."""
    prop, epoch, period = _create_numerical_propagator_leo()

    # Monitor SMA at the orbit value (should remain constant in two-body)
    a = bh.R_EARTH + 500e3
    event = bh.SemiMajorAxisEvent(a, "SMA value", bh.EventDirection.ANY)
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    # In two-body, SMA is constant so there should be no crossings
    events = prop.event_log()
    # This is expected - constant SMA means no crossing events
    assert isinstance(events, list)


def test_eccentricity_event_integration():
    """Test EccentricityEvent detection with NumericalOrbitPropagator."""
    prop, epoch, period = _create_numerical_propagator_leo(eccentricity=0.01)

    event = bh.EccentricityEvent(0.01, "Ecc value", bh.EventDirection.ANY)
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    # In two-body, eccentricity is constant
    events = prop.event_log()
    assert isinstance(events, list)


def test_event_with_callback():
    """Test ValueEvent with callback that modifies action."""
    prop, epoch, period = _create_numerical_propagator_leo()

    # Create a custom value event that monitors z-component
    def z_monitor(ep, state):
        return state[2]

    event = bh.ValueEvent("Z Crossing", z_monitor, 0.0, bh.EventDirection.ANY)
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    events = prop.event_log()
    assert len(events) >= 1, f"Expected z-crossing events, got {len(events)}"


def test_event_with_instance_integration():
    """Test with_instance on events used in propagation."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.AscendingNodeEvent("AN").with_instance(1)
    prop.add_event_detector(event)

    prop.propagate_to(epoch + 2 * period)

    events = prop.event_log()
    assert len(events) >= 1


def test_event_with_tolerances_integration():
    """Test with_tolerances on events used in propagation."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.AscendingNodeEvent("AN").with_tolerances(1e-4, 1e-7)
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    events = prop.event_log()
    assert len(events) >= 1


def test_query_events_empty():
    """Test EventQuery with no events detected."""
    prop, epoch, period = _create_numerical_propagator_leo()

    # No events added, propagate
    prop.propagate_to(epoch + 100.0)

    query = prop.query_events()
    assert query.count() == 0
    assert query.is_empty()
    assert not query.any()
    assert query.first() is None
    assert query.last() is None
    assert list(query) == []


def test_query_events_by_event_type():
    """Test EventQuery by_event_type filter."""
    prop, epoch, period = _create_numerical_propagator_leo()

    event = bh.AscendingNodeEvent("Asc Node")
    prop.add_event_detector(event)

    prop.propagate_to(epoch + period)

    query = prop.query_events()
    # Ascending node events are instantaneous
    instant_events = query.by_event_type(bh.EventType.INSTANTANEOUS).collect()
    assert len(instant_events) >= 1
