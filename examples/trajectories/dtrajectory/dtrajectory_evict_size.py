# /// script
# dependencies = ["brahe"]
# ///
"""
Memory management with maximum size eviction policy
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory with max size limit
traj = bh.DTrajectory(6).with_eviction_policy_max_size(3)

epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Add 5 states
for i in range(5):
    epoch = epoch0 + i * 60.0
    state = np.array([bh.R_EARTH + 500e3 + i * 1000, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch, state)

# Only the 3 most recent states are kept
print(f"Trajectory length: {len(traj)}")
print(f"Start epoch: {traj.start_epoch()}")
print(f"Start altitude: {(traj.state_at_idx(0)[0] - bh.R_EARTH) / 1e3:.2f} km")
# Output: ~502 km (states 0 and 1 were evicted)

# Output
# Trajectory length: 3
# Start epoch: 2024-01-01 00:02:00.000 UTC
# Start altitude: 502.00 km
