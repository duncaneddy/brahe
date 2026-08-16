"""Event detection for numerical orbit propagation.

This module provides event detection capabilities for use with numerical
orbit propagators. Events can monitor various conditions and trigger actions
during propagation.

Event Types:
    - TimeEvent: Detects when simulation time reaches a target epoch
    - ValueEvent: Detects when a monitored value crosses a target value
    - BinaryEvent: Detects boolean condition transitions
    - AltitudeEvent: Detects geodetic altitude crossings (convenience wrapper)

Example:
    ```python
    import brahe as bh
    import numpy as np

    # Simple time event
    event = bh.TimeEvent(target_epoch, "Maneuver Start")

    # Custom value event with value function
    def altitude_fn(epoch, state):
        r = np.linalg.norm(state[:3])
        return r - bh.R_EARTH

    event = bh.ValueEvent(
        "Low Altitude",
        altitude_fn,
        300e3,
        bh.EventDirection.DECREASING
    )
    ```
"""

from brahe._brahe import (
    # Premade event detectors
    AltitudeEvent,
    # AOI (Area of Interest) events
    AOIEntryEvent,
    AOIExitEvent,
    ArgumentOfLatitudeEvent,
    ArgumentOfPerigeeEvent,
    # Node crossing events
    AscendingNodeEvent,
    BinaryEvent,
    DescendingNodeEvent,
    DetectedEvent,
    EccentricAnomalyEvent,
    EccentricityEvent,
    EclipseEvent,
    EdgeType,
    EventAction,
    EventDirection,
    EventQuery,
    EventType,
    InclinationEvent,
    LatitudeEvent,
    LongitudeEvent,
    MeanAnomalyEvent,
    PenumbraEvent,
    # Orbital element events
    SemiMajorAxisEvent,
    # State-derived events
    SpeedEvent,
    SunlitEvent,
    TimeEvent,
    TrueAnomalyEvent,
    # Eclipse/shadow events
    UmbraEvent,
    ValueEvent,
)

__all__ = [
    # AOI (Area of Interest) events
    "AOIEntryEvent",
    "AOIExitEvent",
    # Premade event detectors
    "AltitudeEvent",
    "ArgumentOfLatitudeEvent",
    "ArgumentOfPerigeeEvent",
    # Node crossing events
    "AscendingNodeEvent",
    "BinaryEvent",
    "DescendingNodeEvent",
    "DetectedEvent",
    "EccentricAnomalyEvent",
    "EccentricityEvent",
    "EclipseEvent",
    "EdgeType",
    "EventAction",
    "EventDirection",
    "EventQuery",
    "EventType",
    "InclinationEvent",
    "LatitudeEvent",
    "LongitudeEvent",
    "MeanAnomalyEvent",
    "PenumbraEvent",
    # Orbital element events
    "SemiMajorAxisEvent",
    # State-derived events
    "SpeedEvent",
    "SunlitEvent",
    "TimeEvent",
    "TrueAnomalyEvent",
    # Eclipse/shadow events
    "UmbraEvent",
    "ValueEvent",
]
