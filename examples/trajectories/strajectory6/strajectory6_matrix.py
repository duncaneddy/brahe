# /// script
# dependencies = ["brahe"]
# ///
"""
Convert trajectory data to matrix format for analysis
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory
traj = bh.STrajectory6()
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

for i in range(3):
    epoch = epoch0 + i * 60.0
    state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0 + i * 10, 0.0])
    traj.add(epoch, state)

# Convert to matrix (rows are states, columns are dimensions)
matrix = traj.to_matrix()
print(f"Matrix shape: {matrix.shape}")
# Matrix shape: (3, 6)

print(f"First state velocity: {matrix[0, 4]:.1f} m/s")
# First state velocity: 7600.0 m/s

print(f"Last state velocity: {matrix[2, 4]:.1f} m/s")
# Last state velocity: 7620.0 m/s
