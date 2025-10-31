# /// script
# dependencies = ["brahe"]
# ///
"""
Create a 6D trajectory from existing data
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create epochs
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
epoch1 = epoch0 + 60.0  # 1 minute later
epoch2 = epoch0 + 120.0  # 2 minutes later

# Create 6D states (position + velocity in meters and m/s)
# Each row is one state vector
states = np.array(
    [
        [bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0],
        [bh.R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0],
        [bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, -7600.0, 0.0],
    ]
)

# Create trajectory from data
epochs = [epoch0, epoch1, epoch2]
traj = bh.STrajectory6.from_data(epochs, states)

print(f"Trajectory length: {len(traj)}")
# Trajectory length: 3
