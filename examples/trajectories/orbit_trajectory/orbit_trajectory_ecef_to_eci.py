# /// script
# dependencies = ["brahe"]
# ///
"""
Convert trajectory from ECEF to ECI frame
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory in ECEF frame
traj_ecef = bh.OrbitTrajectory(
    bh.OrbitFrame.ECEF, bh.OrbitRepresentation.CARTESIAN, None
)

# Add dummy states in ECEF
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
for i in range(3):
    epoch = epoch0 + i * 60.0
    # Define state at epoch
    state_ecef = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 7600.0])
    traj_ecef.add(epoch, state_ecef)

print(f"Original frame: {traj_ecef.frame}")  # Output: OrbitFrame.ECEF

# Convert to ECI
traj_eci = traj_ecef.to_eci()

print(f"Converted frame: {traj_eci.frame}")  # Output: OrbitFrame.ECI
print(f"Trajectory length: {len(traj_eci)}")  # Output: 3

# Iterate over converted states
for epoch, state_eci in traj_eci:
    pos_mag = np.linalg.norm(state_eci[0:3])
    vel_mag = np.linalg.norm(state_eci[3:6])
    print(f"Epoch: {epoch}")
    print(f"  Position magnitude: {pos_mag / 1e3:.2f} km")
    print(f"  Velocity magnitude: {vel_mag:.2f} m/s")
