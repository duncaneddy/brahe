# Event Detection

The numerical propagator includes an event detection system that identifies specific orbital conditions during propagation. Events can trigger callbacks for maneuvers, logging, or other actions.

## Event Types

Brahe provides four event detector types:

| Type | Trigger Condition |
|------|-------------------|
| `TimeEvent` | Specific epoch reached |
| `ValueEvent` | Computed quantity crosses threshold |
| `BinaryEvent` | Boolean condition changes |
| `AltitudeEvent` | Spacecraft altitude crosses threshold |

## Time Events

Time events trigger at specific epochs. They're useful for scheduled operations like data collection windows, communication passes, or timed maneuvers.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/event_time.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/event_time.rs:4"
    ```

## Value Events

Value events trigger when a user-defined function crosses a threshold value. This is the most flexible event type, enabling detection of arbitrary orbital conditions.

### Event Direction

Value events can detect:

- `INCREASING` - Value crosses threshold from below
- `DECREASING` - Value crosses threshold from above
- `ANY` - Any threshold crossing

### Custom Value Functions

The value function receives the current epoch and state vector, returning a scalar value:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/custom_value_event.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/custom_value_event.rs:4"
    ```

### Common Value Event Applications

| Event | Value Function | Threshold |
|-------|----------------|-----------|
| Equator crossing | ECI Z-coordinate | 0.0 |
| Apogee/Perigee | Radial velocity | 0.0 |
| Specific longitude | Geodetic longitude | Target value |
| Eclipse entry/exit | Sun angle | Shadow boundary |

## Binary Events

Binary events detect when a boolean condition transitions between true and false. They use `EdgeType` to specify which transition to detect:

- `RISING_EDGE` - Condition becomes true (false → true)
- `FALLING_EDGE` - Condition becomes false (true → false)
- `ANY_EDGE` - Either transition

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/event_binary.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/event_binary.rs:4"
    ```

## Altitude Events

Altitude events detect when the spacecraft crosses a specified geodetic altitude. They're useful for atmospheric re-entry, orbit raising, and altitude-based operations.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/event_altitude.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/event_altitude.rs:4"
    ```

For more details on altitude events and other built-in event types, see [Premade Events](premade_events.md).

## Adding Event Detectors

Event detectors are added to the propagator before propagation:

```python
prop = bh.NumericalOrbitPropagator(...)
prop.add_event_detector(event1)
prop.add_event_detector(event2)
prop.propagate_to(end_epoch)
```

Multiple detectors can be active simultaneously.

## Accessing Event Results

After propagation, detected events are available via the event log:

```python
events = prop.event_log()
for event in events:
    print(f"Event '{event.name}' at {event.window_open}")
    print(f"  State: {event.entry_state}")
```

Each event record contains:

| Field | Description |
|-------|-------------|
| `name` | Event detector name |
| `window_open` | Epoch when event occurred |
| `window_close` | Same as window_open for instantaneous events |
| `entry_state` | State vector at event time |

## Event Detection Algorithm

The propagator uses bisection to locate events precisely:

1. During each integration step, event functions are evaluated
2. Sign changes indicate an event crossing
3. Bisection narrows the event time to within the tolerance
4. The event is recorded with interpolated state

The event detection tolerance is controlled by the integrator configuration. Tighter tolerances provide more precise event times but increase computation.

---

## See Also

- [Numerical Propagation Overview](index.md) - Architecture and concepts
- [Premade Events](premade_events.md) - Built-in event types
- [Event Callbacks](event_callbacks.md) - Responding to detected events
- [Maneuvers](maneuvers.md) - Using events for orbit maneuvers
- [Numerical Orbit Propagator](basic_propagation.md) - Propagator fundamentals
