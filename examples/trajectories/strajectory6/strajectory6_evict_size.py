# /// script
# dependencies = ["brahe"]
# ///
"""
Use maximum size eviction policy to limit trajectory size
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory with max size limit
traj = bh.STrajectory6().with_eviction_policy_max_size(3)

epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Add 5 states
for i in range(5):
    epoch = epoch0 + i * 60.0
    state = np.array([bh.R_EARTH + 500e3 + i * 1000, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch, state)

# Only the 3 most recent states are kept
print(f"Trajectory length: {len(traj)}")
# Trajectory length: 3

print(f"Start altitude: {(traj.state_at_idx(0)[0] - bh.R_EARTH) / 1e3:.2f} km")
# Start altitude: 502.00 km (states 0 and 1 were evicted)
