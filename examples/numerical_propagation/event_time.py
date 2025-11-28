# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using TimeEvent for scheduled event detection.
Demonstrates triggering events at specific times during propagation.
"""

import numpy as np
import brahe as bh

# Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
bh.initialize_eop()
bh.initialize_sw()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
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

# Add time events at specific epochs
event_30min = bh.TimeEvent(epoch + 1800.0, "30-minute mark")
event_1hr = bh.TimeEvent(epoch + 3600.0, "1-hour mark")

# Add a terminal event that stops propagation
event_terminal = bh.TimeEvent(epoch + 5400.0, "90-minute stop").set_terminal()

prop.add_event_detector(event_30min)
prop.add_event_detector(event_1hr)
prop.add_event_detector(event_terminal)

# Propagate for 2 hours (will stop at 90 minutes due to terminal event)
prop.propagate_to(epoch + 7200.0)

# Check detected events
events = prop.event_log()
print(f"Detected {len(events)} events:")
for event in events:
    dt = event.window_open - epoch
    print(f"  '{event.name}' at t+{dt:.1f}s")

# Verify propagation stopped at terminal event
final_time = prop.current_epoch - epoch
print(f"\nPropagation stopped at: t+{final_time:.1f}s (requested: t+7200s)")

# Validate
assert len(events) == 3  # All three events detected
assert abs(final_time - 5400.0) < 1.0  # Stopped at 90 min

print("\nExample validated successfully!")
