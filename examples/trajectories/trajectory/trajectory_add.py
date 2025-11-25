# /// script
# dependencies = ["brahe"]
# ///
"""
Add states to a Trajectory one at a time
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create empty trajectory
traj = bh.Trajectory(6)

# Add states
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
state0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
traj.add(epoch0, state0)

print(f"Trajectory length: {len(traj)}")
# Trajectory length: 1

epoch1 = epoch0 + 60.0
state1 = np.array([bh.R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0])
traj.add(epoch1, state1)

print(f"Trajectory length: {len(traj)}")
# Trajectory length: 2
