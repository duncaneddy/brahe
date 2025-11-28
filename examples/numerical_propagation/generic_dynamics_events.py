# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Event detection with NumericalPropagator.
Demonstrates detecting zero crossings in a simple harmonic oscillator.
"""

import numpy as np
import brahe as bh

# Initialize EOP data (needed for epoch operations)
bh.initialize_eop()

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Simple Harmonic Oscillator
# State: [x, v] where x is position and v is velocity
omega = 2.0 * np.pi  # 1 Hz oscillation frequency

# Initial state: displaced from equilibrium
x0 = 1.0  # 1 meter displacement
v0 = 0.0  # Starting from rest
initial_state = np.array([x0, v0])


def sho_dynamics(t, state, params):
    """Simple harmonic oscillator dynamics."""
    x, v = state[0], state[1]
    omega_sq = params[0] if params is not None else omega**2
    return np.array([v, -omega_sq * x])


# Parameters (omega^2)
params = np.array([omega**2])

# Create propagator
prop = bh.NumericalPropagator(
    epoch,
    initial_state,
    sho_dynamics,
    bh.NumericalPropagationConfig.default(),
    params,
)


# Define value function for zero crossing detection
# ValueEvent receives (epoch, state) and returns a scalar
def position_value(current_epoch, state):
    """Return position component for event detection."""
    return state[0]


# Create ValueEvent to detect position zero crossings
# INCREASING: x goes from negative to positive (moving right through origin)
positive_crossing = bh.ValueEvent(
    "Positive Crossing",
    position_value,
    0.0,  # Target value
    bh.EventDirection.INCREASING,
)

# DECREASING: x goes from positive to negative (moving left through origin)
negative_crossing = bh.ValueEvent(
    "Negative Crossing",
    position_value,
    0.0,
    bh.EventDirection.DECREASING,
)

# Add event detectors to propagator
prop.add_event_detector(positive_crossing)
prop.add_event_detector(negative_crossing)

# Propagate for 5 periods
period = 2 * np.pi / omega  # Period = 1 second
prop.propagate_to(epoch + 5 * period)

# Get event log
events = prop.event_log()

print("Simple Harmonic Oscillator Zero Crossings:")
print(f"  omega = {omega:.4f} rad/s (1 Hz)")
print(f"  Period = {period:.4f} s")
print("  Expected crossings per period: 2 (one each direction)")
print()

positive_events = [e for e in events if "Positive" in e.name]
negative_events = [e for e in events if "Negative" in e.name]

print(f"Total events detected: {len(events)}")
print(f"  Positive crossings: {len(positive_events)}")
print(f"  Negative crossings: {len(negative_events)}")
print()

print("Event details:")
print("  Time (s)   Type               Position     Velocity")
print("-" * 60)

for event in events[:10]:  # Show first 10 events
    t = event.window_open - epoch
    x = event.entry_state[0]
    v = event.entry_state[1]
    print(f"  {t:.4f}     {event.name:<18} {x:+.6f}   {v:+.6f}")

# Validate
# In 5 periods, we should have 5 positive crossings and 5 negative crossings
assert len(positive_events) == 5, (
    f"Expected 5 positive crossings, got {len(positive_events)}"
)
assert len(negative_events) == 5, (
    f"Expected 5 negative crossings, got {len(negative_events)}"
)

# Check timing: crossings should occur at quarter periods
# Starting from x=1, v=0: oscillator moves left first (cosine motion)
# Negative crossing (moving left) at T/4, 5T/4, 9T/4, ...
# Positive crossing (moving right) at 3T/4, 7T/4, 11T/4, ...
expected_negative_times = [(0.25 + i) * period for i in range(5)]
expected_positive_times = [(0.75 + i) * period for i in range(5)]

for i, event in enumerate(negative_events):
    t = event.window_open - epoch
    expected = expected_negative_times[i]
    error = abs(t - expected)
    assert error < 0.02, (
        f"Negative crossing {i}: expected t={expected:.4f}, got t={t:.4f}"
    )

for i, event in enumerate(positive_events):
    t = event.window_open - epoch
    expected = expected_positive_times[i]
    error = abs(t - expected)
    assert error < 0.02, (
        f"Positive crossing {i}: expected t={expected:.4f}, got t={t:.4f}"
    )

print("\nTiming verified: all crossings within 0.02s of expected times")

print("\nExample validated successfully!")
