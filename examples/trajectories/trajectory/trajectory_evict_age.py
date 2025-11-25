# /// script
# dependencies = ["brahe"]
# ///
"""
Memory management with maximum age eviction policy
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Keep only states within last 2 minutes (120 seconds)
traj = bh.Trajectory(6).with_eviction_policy_max_age(120.0)

epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Add states spanning 4 minutes
for i in range(5):
    epoch = epoch0 + i * 60.0
    state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch, state)

# Only states within 120 seconds of the most recent are kept
print(f"Trajectory length: {len(traj)}")
print(f"Timespan: {traj.timespan():.1f} seconds")

# Output:
# Trajectory length: 3
# Timespan: 120.0 seconds
