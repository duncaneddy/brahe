# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using orbital element events to detect inclination value crossings.
Demonstrates InclinationEvent with the angle_format parameter.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state - orbit with inclination near value
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
# SSO-like orbit
oe = np.array([bh.R_EARTH + 600e3, 0.001, 97.8, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

# Create propagator
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.default(),
    params,
)

# Add orbital element events
# Detect when inclination crosses 97.79 degrees (monitoring for stability)
inc_event = bh.InclinationEvent(
    97.79,  # value in degrees
    "Inc value",
    bh.EventDirection.ANY,
    bh.AngleFormat.DEGREES,
)

# Detect semi-major axis value (orbit decay monitoring)
sma_event = bh.SemiMajorAxisEvent(
    bh.R_EARTH + 599.5e3,  # value in meters
    "SMA value",
    bh.EventDirection.DECREASING,
)

prop.add_event_detector(inc_event)
prop.add_event_detector(sma_event)

# Propagate for 3 orbits
orbital_period = 2 * np.pi * np.sqrt(oe[0] ** 3 / bh.GM_EARTH)
prop.propagate_to(epoch + 3 * orbital_period)

# Check detected events
events = prop.event_log()
print(f"Detected {len(events)} orbital element events:")

for event in events:
    dt = event.window_open - epoch
    # Get current orbital elements
    r = event.entry_state[:3]
    v = event.entry_state[3:]
    alt = np.linalg.norm(r) - bh.R_EARTH
    print(f"  '{event.name}' at t+{dt:.1f}s (altitude: {alt / 1e3:.1f} km)")

# Count events by type
inc_events = [e for e in events if "Inc" in e.name]
sma_events = [e for e in events if "SMA" in e.name]

print(f"\nInclination value crossings: {len(inc_events)}")
print(f"SMA value crossings: {len(sma_events)}")

# The J2 perturbation causes slow variations - we may or may not cross values
# depending on the exact parameters, so we just validate the events work
print("\nExample completed successfully!")
