# /// script
# dependencies = ["brahe"]
# ///
"""
Retrieve states and epochs by their index
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create and populate trajectory
traj = bh.Trajectory(6)
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
state0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
traj.add(epoch0, state0)

epoch1 = epoch0 + 60.0
state1 = np.array([bh.R_EARTH + 600e3, 456000.0, 0.0, -7600.0, 0.0, 0.0])
traj.add(epoch1, state1)

epoch2 = epoch0 + 120.0
state2 = np.array([bh.R_EARTH + 700e3, 0.0, 0.0, 0.0, -7600.0, 0.0])
traj.add(epoch2, state2)

# Access by index
retrieved_epoch = traj.epoch_at_idx(1)
retrieved_state = traj.state_at_idx(1)

print(f"Epoch: {retrieved_epoch}")
print(f"Altitude: {retrieved_state[0] - bh.R_EARTH:.2f} m")

# Output:
# Epoch: 2024-01-01 00:01:00.000 UTC
# Altitude: 600000.00 m
