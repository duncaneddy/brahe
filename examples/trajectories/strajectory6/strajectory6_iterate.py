# /// script
# dependencies = ["brahe"]
# ///
"""
Iterate over all epoch-state pairs in a trajectory
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create and populate trajectory
traj = bh.STrajectory6()
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

for i in range(3):
    epoch = epoch0 + i * 60.0
    state = np.array([bh.R_EARTH + 500e3 + i * 1000, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch, state)

# Iterate over all states
for epoch, state in traj:
    altitude = (state[0] - bh.R_EARTH) / 1e3
    velocity = np.linalg.norm(state[3:6])
    print(f"Epoch: {epoch}, Altitude: {altitude:.2f} km, Speed: {velocity:.0f} m/s")
# Epoch: 2024-01-01 00:00:00.000 UTC, Altitude: 500.00 km, Speed: 7600 m/s
# Epoch: 2024-01-01 00:01:00.000 UTC, Altitude: 501.00 km, Speed: 7600 m/s
# Epoch: 2024-01-01 00:02:00.000 UTC, Altitude: 502.00 km, Speed: 7600 m/s
