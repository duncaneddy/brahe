# /// script
# dependencies = ["brahe"]
# ///
"""
Create an empty 6D trajectory
"""

import brahe as bh

bh.initialize_eop()

# Create empty 6D trajectory
traj = bh.STrajectory6()
print(f"Trajectory length: {len(traj)}")
# Trajectory length: 0

print(f"Is empty: {traj.is_empty()}")
# Is empty: True
