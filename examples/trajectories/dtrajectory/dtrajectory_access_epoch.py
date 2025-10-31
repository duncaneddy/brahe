# /// script
# dependencies = ["brahe"]
# ///
"""
Get states at or near specific epochs
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory with multiple states
traj = bh.DTrajectory(6)
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

for i in range(5):
    epoch = epoch0 + i * 60.0
    state = np.array([bh.R_EARTH + 500e3 + i * 1000, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch, state)

# Get nearest state to a specific epoch
query_epoch = epoch0 + 120.0  # 2 minutes after start
nearest_epoch, nearest_state = traj.nearest_state(query_epoch)
print(
    f"Nearest state at t+120s altitude: {(nearest_state[0] - bh.R_EARTH) / 1e3:.2f} km"
)

# Get nearest state between stored epochs
query_epoch = epoch0 + 125.0  # Between stored epochs
nearest_epoch, nearest_state = traj.nearest_state(query_epoch)
print(
    f"Nearest state at t+125s altitude: {(nearest_state[0] - bh.R_EARTH) / 1e3:.2f} km"
)

# Output:
# Nearest state at t+120s altitude: 502.00 km
# Nearest state at t+125s altitude: 502.00 km
