# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using eclipse events to detect shadow transitions.
Demonstrates detecting when a spacecraft enters or exits Earth's shadow.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state - LEO orbit
epoch = bh.Epoch.from_datetime(2024, 6, 21, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
# LEO orbit with some inclination
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

# Add eclipse events with different edge types
# Detect entry into eclipse (any shadow - umbra or penumbra)
eclipse_entry = bh.EclipseEvent("Eclipse Entry", bh.EdgeType.RISING_EDGE, None)

# Detect exit from eclipse
eclipse_exit = bh.EclipseEvent("Eclipse Exit", bh.EdgeType.FALLING_EDGE, None)

prop.add_event_detector(eclipse_entry)
prop.add_event_detector(eclipse_exit)

# Propagate for 5 orbits
orbital_period = bh.orbital_period(oe[0])
prop.propagate_to(epoch + 5 * orbital_period)

# Check detected events
events = prop.event_log()
print(f"Detected {len(events)} eclipse events:")

for event in events:
    dt = event.window_open - epoch
    print(f"  '{event.name}' at t+{dt:.1f}s")

# Count events by type
entries = [e for e in events if "Entry" in e.name]
exits = [e for e in events if "Exit" in e.name]

print(f"\nEclipse entries: {len(entries)}")
print(f"Eclipse exits: {len(exits)}")

# Calculate eclipse durations
if len(entries) > 0 and len(exits) > 0:
    # Find pairs of entry/exit events
    durations = []
    for i, entry in enumerate(entries):
        # Find next exit after this entry
        for exit_event in exits:
            if exit_event.window_open > entry.window_open:
                duration = exit_event.window_open - entry.window_open
                durations.append(duration)
                break

    if durations:
        avg_duration = sum(durations) / len(durations)
        print(
            f"\nAverage eclipse duration: {avg_duration:.1f}s ({avg_duration / 60:.1f} min)"
        )

# Validate - should have roughly equal entries and exits
assert abs(len(entries) - len(exits)) <= 1, "Entry/exit count mismatch"
assert len(entries) >= 4, "Expected at least 4 eclipse entries in 5 orbits"

print("\nExample validated successfully!")
