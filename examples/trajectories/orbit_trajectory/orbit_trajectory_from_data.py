# /// script
# dependencies = ["brahe"]
# ///
"""
Create SOrbitTrajectory from existing epochs and orbital states
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create epochs
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
epoch1 = epoch0 + 60.0
epoch2 = epoch0 + 120.0

# Create Cartesian states in ECI
state0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
state1 = np.array([bh.R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0])
state2 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, -7600.0, 0.0])

# Create trajectory from data
epochs = [epoch0, epoch1, epoch2]
states = np.array([state0, state1, state2])  # Flattened array
traj = bh.SOrbitTrajectory.from_orbital_data(
    epochs, states, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None
)

print(f"Trajectory length: {len(traj)}")
# Trajectory length: 3
