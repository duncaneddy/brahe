# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Numerical propagation with trajectory history access.
Demonstrates accessing stored trajectory states and interpolation.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Define orbital elements: LEO satellite
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Parameters: [mass, drag_area, Cd, srp_area, Cr]
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

# Create propagator
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.default(),
    params,
)

# Propagate for 2 hours
prop.propagate_to(epoch + 7200.0)

# Access trajectory
trajectory = prop.trajectory

# Get trajectory length (number of stored points)
num_points = len(trajectory)
print(f"Trajectory points stored: {num_points}")

# Get state at intermediate time using interpolation
mid_time = epoch + 3600.0  # 1 hour in
mid_state = prop.state(mid_time)
print(f"State at t+1h: position norm = {np.linalg.norm(mid_state[:3]) / 1e3:.3f} km")

# Get states at multiple times
times = [epoch + t for t in [1800.0, 3600.0, 5400.0, 7200.0]]
for t in times:
    s = prop.state(t)
    alt = np.linalg.norm(s[:3]) - bh.R_EARTH
    print(f"  t+{(t - epoch):.0f}s: altitude = {alt / 1e3:.3f} km")

# Get state in different frames
ecef_state = prop.state_ecef(mid_time)
koe = prop.state_koe(mid_time, bh.AngleFormat.DEGREES)
print(
    f"\nECEF position (km): [{ecef_state[0] / 1e3:.3f}, {ecef_state[1] / 1e3:.3f}, {ecef_state[2] / 1e3:.3f}]"
)
print(
    f"Keplerian elements: a={koe[0] / 1e3:.3f} km, e={koe[1]:.6f}, i={koe[2]:.2f} deg"
)

# Validate
assert num_points > 1
assert len(mid_state) == 6
assert np.linalg.norm(mid_state[:3]) > bh.R_EARTH

print("\nExample validated successfully!")
