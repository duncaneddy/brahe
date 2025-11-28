# Event Detection

The numerical propagator includes an event detection system that identifies specific orbital conditions during propagation. Events are defined by user-configurable detectors that monitor the spacecraft state and trigger when certain criteria are met. They can also be coupled with event callbacks to respond to detected events in real-time.

When an event is detected, the propagator uses a bisection algorithm to precisely locate the event time within a specified tolerance. The detected events are logged and can be accessed after propagation. Users can also configure how an event will affect the propagation, such as stopping propagation or continuing without interruption.

Events provide an extensible mechanism for implementing complex mission scenarios, such as maneuver execution, autonomous operations, and other condition-based actions.

The library also provides a set of premade event detectors for common scenarios, which can be used directly or serve as templates for custom detectors. You can find more details about premade events in the [Premade Events](premade_events.md) documentation with a complete list of available types in the library API docuementation at [Premade Event Detectors](../../../library_api/events/premade.md).

## Event Types

Brahe provides three event fundamental event detector types:

<div class="center-table" markdown="1">
| Type | Trigger Condition |
|------|-------------------|
| `TimeEvent` | Specific epoch reached |
| `ValueEvent` | Computed quantity crosses a given value |
| `BinaryEvent` | Boolean condition changes |
</div>


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

<div class="center-table" markdown="1">
| Field | Description |
|-------|-------------|
| `name` | Event detector name |
| `window_open` | Epoch when event occurred |
| `window_close` | Same as window_open for instantaneous events |
| `entry_state` | State vector at event time |
</div>


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

Value events trigger when a user-defined function crosses a value value. This is the most flexible event type, enabling detection of arbitrary orbital conditions.

Value events are defined with a value function which accepts the current epoch and state vector, returning a scalar value.

### Event Direction

Value events can detect:

- `INCREASING` - Value crosses value from below
- `DECREASING` - Value crosses value from above
- `ANY` - Any value crossing

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

## Binary Events

Binary events detect when a boolean condition transitions between true and false. They use `EdgeType` to specify which transition to detect:

- `RISING_EDGE` - Condition becomes true (false → true)
- `FALLING_EDGE` - Condition becomes false (true → false)
- `ANY_EDGE` - Either transition

The binary condition function receives the current epoch and state vector, returning a boolean value.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/event_binary.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/event_binary.rs:4"
    ```

---

## See Also

- [Numerical Propagation Overview](index.md) - Architecture and concepts
- [Premade Events](premade_events.md) - Built-in event types
- [Event Callbacks](event_callbacks.md) - Responding to detected events
- [Maneuvers](maneuvers.md) - Using events for orbit maneuvers
- [Numerical Orbit Propagator](numerical_orbit_propagator.md) - Propagator fundamentals
