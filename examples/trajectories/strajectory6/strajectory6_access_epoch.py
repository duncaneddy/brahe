# /// script
# dependencies = ["brahe"]
# ///
"""
Access trajectory states at specific epochs
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory with multiple states
traj = bh.STrajectory6()
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

for i in range(5):
    epoch = epoch0 + i * 60.0
    state = np.array([bh.R_EARTH + 500e3 + i * 1000, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch, state)

# Get nearest state (exact match)
query_epoch = epoch0 + 120.0  # 2 minutes after start
nearest_epoch, nearest_state = traj.nearest_state(query_epoch)
print(f"Exact match found at altitude: {(nearest_state[0] - bh.R_EARTH) / 1e3:.2f} km")

# Get nearest state (between stored epochs)
query_epoch = epoch0 + 125.0  # Between stored epochs
nearest_epoch, nearest_state = traj.nearest_state(query_epoch)
print(f"Nearest state altitude: {(nearest_state[0] - bh.R_EARTH) / 1e3:.2f} km")

# Output:
# Exact match found at altitude: 502.00 km
# Nearest state altitude: 502.00 km
