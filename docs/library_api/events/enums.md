# Enumerations

Enumerations used by the event detection system.

## EventDirection

Specifies which type of zero-crossing to detect for value-based events.

::: brahe.EventDirection
    options:
      show_root_heading: true
      show_root_full_path: false

## EventAction

Action to take when an event is detected (stop or continue propagation).

::: brahe.EventAction
    options:
      show_root_heading: true
      show_root_full_path: false

## EventType

Type of event: instantaneous (single point in time) or period (maintains condition over interval).

::: brahe.EventType
    options:
      show_root_heading: true
      show_root_full_path: false

## EdgeType

Edge type for binary event detection (rising, falling, or any edge).

::: brahe.EdgeType
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [Event Detectors](detectors.md) - Core event detector classes
- [Pre-made Events](premade.md) - Convenience event detectors
- [Event Results](results.md) - DetectedEvent and EventQuery
