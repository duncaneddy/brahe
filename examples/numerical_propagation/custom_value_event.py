# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Creating custom ValueEvent detectors.
Demonstrates detecting when a computed value crosses a value.
"""

import numpy as np
import brahe as bh

# Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
bh.initialize_eop()
bh.initialize_sw()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 15.0, 30.0, 45.0])
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


# Define custom value function: detect when z-component crosses zero
# This detects equator crossings (ascending and descending node)
def z_position(epoch, state):
    """Return z-component of position (meters)."""
    return state[2]


# Create ValueEvent: detect when z crosses 0 (equator crossing)
# Ascending node: z goes from negative to positive (INCREASING)
ascending_node = bh.ValueEvent(
    "Ascending Node",
    z_position,
    0.0,  # target value
    bh.EventDirection.INCREASING,
)

# Descending node: z goes from positive to negative (DECREASING)
descending_node = bh.ValueEvent(
    "Descending Node",
    z_position,
    0.0,
    bh.EventDirection.DECREASING,
)

prop.add_event_detector(ascending_node)
prop.add_event_detector(descending_node)

# Propagate for 3 orbits
orbital_period = bh.orbital_period(oe[0])
prop.propagate_to(epoch + 3 * orbital_period)

# Check detected events
events = prop.event_log()
ascending = [e for e in events if "Ascending" in e.name]
descending = [e for e in events if "Descending" in e.name]

print("Equator crossings over 3 orbits:")
print(f"  Ascending nodes: {len(ascending)}")
print(f"  Descending nodes: {len(descending)}")

for event in events[:6]:  # Show first 6
    dt = event.window_open - epoch
    z = event.entry_state[2]
    print(f"  '{event.name}' at t+{dt:.1f}s (z={z:.1f} m)")

# Validate
assert len(ascending) == 3  # One per orbit
assert len(descending) == 3  # One per orbit

print("\nExample validated successfully!")
