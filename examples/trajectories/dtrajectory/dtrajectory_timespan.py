# /// script
# dependencies = ["brahe"]
# ///
"""
Query the temporal extent of a trajectory
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory spanning 5 minutes
traj = bh.DTrajectory(6)
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

for i in range(6):
    epoch = epoch0 + i * 60.0
    state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch, state)

# Query properties
print(f"Number of states: {len(traj)}")
print(f"Start epoch: {traj.start_epoch()}")
print(f"End epoch: {traj.end_epoch()}")
print(f"Timespan: {traj.timespan():.1f} seconds")
print(f"Is empty: {traj.is_empty()}")

# Access first and last states
first_epoch, first_state = traj.first()
last_epoch, last_state = traj.last()
print(f"First epoch: {first_epoch}")
print(f"Last epoch: {last_epoch}")

# Output:
# Number of states: 6
# Start epoch: 2024-01-01 00:00:00.000 UTC
# End epoch: 2024-01-01 00:05:00.000 UTC
# Timespan: 300.0 seconds
# Is empty: False
# First epoch: 2024-01-01 00:00:00.000 UTC
# Last epoch: 2024-01-01 00:05:00.000 UTC
