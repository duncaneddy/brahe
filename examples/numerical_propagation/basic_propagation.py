# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Basic numerical orbit propagation using NumericalOrbitPropagator.
Demonstrates creating a propagator and propagating to a target time.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Define orbital elements: [a, e, i, raan, argp, M] in SI units
# LEO satellite: 500 km altitude, near-circular, sun-synchronous inclination
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Parameters: [mass, drag_area, Cd, srp_area, Cr]
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

# Create propagator with default configuration
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.default(),
    params,
)

# Propagate for 1 hour
prop.propagate_to(epoch + 3600.0)

# Get final state
final_epoch = prop.current_epoch
final_state = prop.current_state()

# Validate propagation completed
assert final_epoch == epoch + 3600.0
assert len(final_state) == 6
assert np.linalg.norm(final_state[:3]) > bh.R_EARTH  # Still in orbit

print(f"Initial epoch: {epoch}")
print(f"Final epoch:   {final_epoch}")
print(
    f"Position (km): [{final_state[0] / 1e3:.3f}, {final_state[1] / 1e3:.3f}, {final_state[2] / 1e3:.3f}]"
)
print(
    f"Velocity (m/s): [{final_state[3]:.3f}, {final_state[4]:.3f}, {final_state[5]:.3f}]"
)
print("Example validated successfully!")
