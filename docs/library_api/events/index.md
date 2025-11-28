# Event Detection

Event detection system for numerical propagators. Events allow detection of specific conditions during propagation, such as reaching a target time, crossing an altitude value, or triggering on custom conditions.

!!! note
    For conceptual explanations and usage examples, see [Event Detection](../../learn/orbit_propagation/numerical_propagation/event_detection.md) in the User Guide.

## Module Structure

The event detection system is organized into the following components:

- **[Event Detectors](detectors.md)** - Core event detector classes (TimeEvent, ValueEvent, BinaryEvent)
- **[Pre-made Events](premade.md)** - Convenience event detectors for common scenarios (AltitudeEvent)
- **[Event Results](results.md)** - DetectedEvent and EventQuery for accessing event information
- **[Enumerations](enums.md)** - EventDirection, EventAction, EventType, EdgeType

---

## See Also

- [NumericalOrbitPropagator](../propagators/numerical_orbit_propagator.md) - Propagator with event detection
- [NumericalPropagator](../propagators/numerical_propagator.md) - Generic propagator with event detection
- [Event Detection Guide](../../learn/orbit_propagation/numerical_propagation/event_detection.md) - User guide
