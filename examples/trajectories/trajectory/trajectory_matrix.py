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
traj = bh.Trajectory(6)
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

for i in range(3):
    epoch = epoch0 + i * 60.0
    state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0 + i * 10, 0.0])
    traj.add(epoch, state)

# Convert to matrix (rows are states, columns are dimensions)
matrix = traj.to_matrix()
print(f"Matrix type: {type(matrix)}")
print(f"Matrix shape: {matrix.shape}")
print(f"First state velocity: {matrix[0, 4]:.1f} m/s")

# Output:
# Matrix type: <class 'numpy.ndarray'>
# Matrix shape: (3, 6)
# First state velocity: 7600.0 m/s
