# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using AltitudeEvent for altitude value detection.
Demonstrates detecting when altitude crosses a specified value.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state - elliptical orbit
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
# Elliptical orbit: 300 km perigee, 800 km apogee
oe = np.array([bh.R_EARTH + 550e3, 0.036, 45.0, 0.0, 0.0, 0.0])
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

# Add altitude events
# Detect when crossing 500 km altitude (both directions)
event_500km = bh.AltitudeEvent(
    500e3,  # value altitude in meters
    "500km crossing",
    bh.EventDirection.ANY,  # Detect both increasing and decreasing
)

# Detect only when ascending through 600 km
event_600km_up = bh.AltitudeEvent(
    600e3,
    "600km ascending",
    bh.EventDirection.INCREASING,
)

prop.add_event_detector(event_500km)
prop.add_event_detector(event_600km_up)

# Propagate for 2 orbits
orbital_period = 2 * np.pi * np.sqrt(oe[0] ** 3 / bh.GM_EARTH)
prop.propagate_to(epoch + 2 * orbital_period)

# Check detected events
events = prop.event_log()
print(f"Detected {len(events)} altitude events:")

for event in events:
    dt = event.window_open - epoch
    alt = np.linalg.norm(event.entry_state[:3]) - bh.R_EARTH
    print(f"  '{event.name}' at t+{dt:.1f}s (altitude: {alt / 1e3:.1f} km)")

# Count events by type
crossings_500 = [e for e in events if "500km" in e.name]
crossings_600 = [e for e in events if "600km" in e.name]

print(f"\n500 km crossings (any direction): {len(crossings_500)}")
print(f"600 km ascending crossings: {len(crossings_600)}")

# Validate
assert len(crossings_500) >= 4  # At least 2 per orbit, 2 orbits
assert len(crossings_600) >= 2  # At least 1 per orbit (ascending only)

print("\nExample validated successfully!")
