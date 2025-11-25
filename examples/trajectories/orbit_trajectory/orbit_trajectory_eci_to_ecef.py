# /// script
# dependencies = ["brahe"]
# ///
"""
Convert trajectory from ECI to ECEF frame
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory in ECI frame
traj_eci = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)

# Add states in ECI
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
for i in range(5):
    epoch = epoch0 + i * 60.0
    # Define state at epoch
    state_eci = np.array([bh.R_EARTH + 500e3, i * 100e3, 0.0, 0.0, 7600.0, 0.0])
    traj_eci.add(epoch, state_eci)

print(f"Original frame: {traj_eci.frame}")
print(f"Original representation: {traj_eci.representation}")

# Convert all states in trajectory to ECEF
traj_ecef = traj_eci.to_ecef()

print(f"\nConverted frame: {traj_ecef.frame}")
print(f"Converted representation: {traj_ecef.representation}")
print(f"Same number of states: {len(traj_ecef)}")

# Compare first states
_, state_eci = traj_eci.first()
_, state_ecef = traj_ecef.first()
print(
    f"\nFirst ECI state: [{state_eci[0]:.2f}, {state_eci[1]:.2f}, {state_eci[2]:.2f}] m"
)
print(
    f"First ECEF state: [{state_ecef[0]:.2f}, {state_ecef[1]:.2f}, {state_ecef[2]:.2f}] m"
)

# Output:
# Original frame: ECI
# Original representation: Cartesian

# Converted frame: ECEF
# Converted representation: Cartesian
# Same number of states: 5

# First ECI state: [6878136.30, 0.00, 0.00] m
# First ECEF state: [-1176064.06, -6776826.51, 15961.82] m
