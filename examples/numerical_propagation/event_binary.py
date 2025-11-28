# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using BinaryEvent for boolean condition detection.
Demonstrates detecting when a condition transitions between true and false.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)


# Define condition function: check if spacecraft is in northern hemisphere
def in_northern_hemisphere(epoch, state):
    """Returns True if z-position is positive (northern hemisphere)."""
    return state[2] > 0


# Create binary events for hemisphere crossings
# Rising edge: false → true (entering northern hemisphere)
enter_north = bh.BinaryEvent(
    "Enter Northern",
    in_northern_hemisphere,
    bh.EdgeType.RISING_EDGE,
)

# Falling edge: true → false (exiting northern hemisphere)
exit_north = bh.BinaryEvent(
    "Exit Northern",
    in_northern_hemisphere,
    bh.EdgeType.FALLING_EDGE,
)

# Create propagator
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
)

prop.add_event_detector(enter_north)
prop.add_event_detector(exit_north)

# Propagate for 2 orbits
orbital_period = bh.orbital_period(oe[0])
prop.propagate_to(epoch + 2 * orbital_period)

# Check detected events
events = prop.event_log()
enters = [e for e in events if "Enter" in e.name]
exits = [e for e in events if "Exit" in e.name]

print("Hemisphere crossings over 2 orbits:")
print(f"  Entered northern: {len(enters)} times")
print(f"  Exited northern:  {len(exits)} times")

print("\nEvent timeline:")
for event in events[:8]:  # First 8 events
    dt = event.window_open - epoch
    z_km = event.entry_state[2] / 1e3
    print(f"  t+{dt:7.1f}s: {event.name:16} (z = {z_km:+.1f} km)")

# Validate
assert len(enters) == 2  # Once per orbit
assert len(exits) == 2  # Once per orbit

print("\nExample validated successfully!")
