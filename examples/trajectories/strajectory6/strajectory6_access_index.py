# /// script
# dependencies = ["brahe"]
# ///
"""
Access trajectory states and epochs by index
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create and populate trajectory
traj = bh.STrajectory6()
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
state0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
traj.add(epoch0, state0)

epoch1 = epoch0 + 60.0
state1 = np.array([bh.R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0])
traj.add(epoch1, state1)

epoch2 = epoch1 + 60.0
state2 = np.array([bh.R_EARTH + 500e3, 912000.0, 0.0, 0.0, -7600.0, 0.0])
traj.add(epoch2, state2)

# Access by index
retrieved_epoch = traj.epoch_at_idx(1)
retrieved_state = traj.state_at_idx(1)

print(f"Epoch: {retrieved_epoch}")
print(
    f"Position: [{retrieved_state[0]:.2f}, {retrieved_state[1]:.2f}, {retrieved_state[2]:.2f}] m"
)
print(
    f"Velocity: [{retrieved_state[3]:.2f}, {retrieved_state[4]:.2f}, {retrieved_state[5]:.2f}] m/s"
)

# Output:
# Epoch: 2024-01-01 00:01:00.000 UTC
# Position: [6878136.30, 456000.00, 0.00] m
# Velocity: [-7600.00, 0.00, 0.00] m/s
