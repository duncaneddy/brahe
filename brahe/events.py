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
    EventDirection,
    EdgeType,
    EventAction,
    EventType,
    DetectedEvent,
    EventQuery,
    TimeEvent,
    ValueEvent,
    BinaryEvent,
    AltitudeEvent,
)

__all__ = [
    "EventDirection",
    "EdgeType",
    "EventAction",
    "EventType",
    "DetectedEvent",
    "EventQuery",
    "TimeEvent",
    "ValueEvent",
    "BinaryEvent",
    "AltitudeEvent",
]
