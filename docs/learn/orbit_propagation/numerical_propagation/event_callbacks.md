# Event Callbacks

Event callbacks allow you to respond to detected events during propagation. Callbacks can log information, inspect state, modify the spacecraft state (for impulsive maneuvers), or control propagation flow.

## Callback Function Signature

To define a callback, create a function matching the following signature:

=== "Python"

    ```python
    def callback(epoch: Epoch, state: np.ndarray) -> tuple[np.ndarray, EventAction]:
        """
        Args:
            epoch: The epoch when the event occurred
            state: The spacecraft state vector at event time [x, y, z, vx, vy, vz]

        Returns:
            tuple: (new_state, action)
                - new_state: Modified state vector (or original if unchanged)
                - action: EventAction.CONTINUE or EventAction.STOP
        """
        # Process event...
        return (state, bh.EventAction.CONTINUE)
    ```

=== "Rust"

    ```rust
    type DEventCallback = Box<
        dyn Fn(
            Epoch,                           // Event epoch
            &DVector<f64>,                   // Current state
            Option<&DVector<f64>>,           // Optional parameters
        ) -> (
            Option<DVector<f64>>,            // New state (None = unchanged)
            Option<DVector<f64>>,            // New params (None = unchanged)
            EventAction,                     // Continue or Stop
        ) + Send + Sync,
    >;
    ```


### EventAction Options

The callback return value includes an `EventAction` that controls propagation behavior:

<div class="center-table" markdown="1">
| Action | Behavior |
|--------|----------|
| `CONTINUE` | Continue propagation after processing the event |
| `STOP` | Halt propagation immediately after the event |
</div>

#### When to Use STOP

Use `EventAction.STOP` when:

- A terminal condition has been reached (e.g., re-entry)
- The propagation goal has been achieved
- An error condition is detected
- You want to examine state at a specific event before deciding to continue

#### When to Use CONTINUE

Use `EventAction.CONTINUE` for:

- Logging and monitoring events
- Impulsive maneuvers (state changes but propagation continues)
- Intermediate waypoints
- Data collection triggers


## Defining Callbacks

Callbacks receive the event epoch and state, and return a tuple containing the (possibly modified) state and an action directive.

### Logging Callback

A simple callback that logs event information without modifying state:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/event_callback_logging.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/event_callback_logging.rs:4"
    ```

## Attaching Callbacks to Events

Use the `with_callback()` method to attach a callback to any event detector:

```python
# Create event
event = bh.TimeEvent(target_epoch, "My Event")

# Attach callback
event_with_callback = event.with_callback(my_callback_function)

# Add to propagator
prop.add_event_detector(event_with_callback)
```

The `with_callback()` method returns a new event detector with the callback attached, allowing method chaining.

## State Modification

Callbacks can modify the spacecraft state by returning a new state vector. This is the mechanism for implementing impulsive maneuvers.

### Modifying State

```python
def velocity_change_callback(epoch, state):
    new_state = state.copy()

    # Add delta-v in velocity direction
    v = state[3:6]
    v_hat = v / np.linalg.norm(v)
    delta_v = 100.0  # m/s
    new_state[3:6] += delta_v * v_hat

    return (new_state, bh.EventAction.CONTINUE)
```

### Physical Consistency

When modifying state, ensure physical consistency:

- **Position changes** are unusual except for specific scenarios
- **Velocity changes** should respect momentum conservation for realistic maneuvers
- **Large changes** may cause numerical issues in subsequent integration steps

For complete impulsive maneuver examples, see [Maneuvers](maneuvers.md).

## Multiple Callbacks

Each event detector can have one callback. For multiple actions at the same event, either:

1. Perform all actions within a single callback
2. Create multiple event detectors at the same time/condition

```python
# Single callback performing multiple actions
def multi_action_callback(epoch, state):
    log_event(epoch, state)
    record_telemetry(epoch, state)
    new_state = apply_correction(state)
    return (new_state, bh.EventAction.CONTINUE)
```

## Callback Execution Order

When multiple events occur at the same epoch:

1. Events are processed in the order their detectors were added
2. State modifications from earlier callbacks are passed to later callbacks
3. If any callback returns `STOP`, propagation halts after all callbacks execute

---

## See Also

- [Event Detection](event_detection.md) - Event detection fundamentals
- [Premade Events](premade_events.md) - Built-in event types
- [Maneuvers](maneuvers.md) - Using callbacks for orbit maneuvers
