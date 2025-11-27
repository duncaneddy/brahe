# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Event callback examples for logging and state inspection.
Demonstrates defining and attaching callbacks to event detectors.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state - elliptical orbit
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Track callback invocations
callback_count = 0


# Define a logging callback
def logging_callback(event_epoch, event_state):
    """Log event details without modifying state."""
    global callback_count
    callback_count += 1

    # Compute orbital elements at event time
    koe = bh.state_eci_to_koe(event_state, bh.AngleFormat.DEGREES)
    altitude = koe[0] - bh.R_EARTH

    print(f"  Event #{callback_count}:")
    print(f"    Epoch: {event_epoch}")
    print(f"    Altitude: {altitude / 1e3:.1f} km")
    print(f"    True anomaly: {koe[5]:.1f} deg")

    # Return unchanged state with CONTINUE action
    return (event_state, bh.EventAction.CONTINUE)


# Define a callback that stops propagation
def stop_callback(event_epoch, event_state):
    """Stop propagation when event occurs."""
    print(f"  Stopping at {event_epoch}")
    return (event_state, bh.EventAction.STOP)


# Create propagator
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
)

# Create time event with logging callback
event_log = bh.TimeEvent(epoch + 1000.0, "Log Event").with_callback(logging_callback)
prop.add_event_detector(event_log)

# Create another time event
event_log2 = bh.TimeEvent(epoch + 2000.0, "Log Event 2").with_callback(logging_callback)
prop.add_event_detector(event_log2)

# Propagate for half an orbit
orbital_period = 2 * np.pi * np.sqrt(oe[0] ** 3 / bh.GM_EARTH)
print("Propagating with logging callbacks:")
prop.propagate_to(epoch + orbital_period / 2)

print(f"\nCallback invoked {callback_count} times")

# Now demonstrate STOP action
print("\nDemonstrating STOP action:")
prop2 = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
)

# Event that stops propagation at t+500s
stop_event = bh.TimeEvent(epoch + 500.0, "Stop Event").with_callback(stop_callback)
prop2.add_event_detector(stop_event)

# Try to propagate for one full orbit
prop2.propagate_to(epoch + orbital_period)

# Check where propagation actually stopped
actual_duration = prop2.current_epoch - epoch
print(f"  Requested duration: {orbital_period:.1f}s")
print(f"  Actual duration: {actual_duration:.1f}s")
print(f"  Stopped early: {actual_duration < orbital_period}")

# Validate
assert callback_count == 2
assert actual_duration < orbital_period

print("\nExample validated successfully!")
