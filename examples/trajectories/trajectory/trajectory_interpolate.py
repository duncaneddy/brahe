# /// script
# dependencies = ["brahe"]
# ///
"""
Linear interpolation to estimate states at arbitrary epochs
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory with sparse data
traj = bh.Trajectory(6)
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Add states every 60 seconds
for i in range(3):
    epoch = epoch0 + i * 60.0
    # Simplified motion: position changes linearly with time
    state = np.array([bh.R_EARTH + 500e3 + i * 10000, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch, state)

# Interpolate state at intermediate time
query_epoch = epoch0 + 30.0  # Halfway between first two states
interpolated_state = traj.interpolate(query_epoch)

print(f"Interpolated altitude: {(interpolated_state[0] - bh.R_EARTH) / 1e3:.2f} km")
# Expected: approximately 505 km (halfway between 500 and 510 km)

# Output:
# Interpolated altitude: 505.00 km
