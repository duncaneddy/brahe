# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using node crossing events to detect equatorial crossings.
Demonstrates detecting ascending and descending node passages.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state - inclined orbit
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
# Inclined orbit for clear node crossings
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0])
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

# Add node crossing events
# Ascending node: spacecraft crosses equator heading north (argument of latitude = 0)
asc_event = bh.AscendingNodeEvent("Ascending Node")

# Descending node: spacecraft crosses equator heading south (argument of latitude = 180 deg)
desc_event = bh.DescendingNodeEvent("Descending Node")

prop.add_event_detector(asc_event)
prop.add_event_detector(desc_event)

# Propagate for 3 orbits
orbital_period = 2 * np.pi * np.sqrt(oe[0] ** 3 / bh.GM_EARTH)
prop.propagate_to(epoch + 3 * orbital_period)

# Check detected events
events = prop.event_log()
print(f"Detected {len(events)} node crossing events:")

for event in events:
    dt = event.window_open - epoch
    # Compute geodetic latitude at event
    r_eci = event.entry_state[:3]
    r_ecef = bh.position_eci_to_ecef(event.window_open, r_eci)
    geodetic = bh.position_ecef_to_geodetic(r_ecef, bh.AngleFormat.DEGREES)
    lat = geodetic[1]
    print(f"  '{event.name}' at t+{dt:.1f}s (latitude: {lat:.2f} deg)")

# Count events by type
ascending = [e for e in events if "Ascending" in e.name]
descending = [e for e in events if "Descending" in e.name]

print(f"\nAscending node crossings: {len(ascending)}")
print(f"Descending node crossings: {len(descending)}")

# Validate
assert len(ascending) >= 3  # At least 3 ascending in 3 orbits
assert len(descending) >= 3  # At least 3 descending in 3 orbits

print("\nExample validated successfully!")
