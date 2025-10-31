# /// script
# dependencies = ["brahe"]
# ///
"""
Create empty DTrajectory instances with different dimensions
"""

import brahe as bh

bh.initialize_eop()

# Create 6D trajectory (default)
traj = bh.DTrajectory()
print(f"Dimension: {traj.dimension()}")
# Dimension: 6

# Create 3D trajectory (position only)
traj_3d = bh.DTrajectory(3)
print(f"Dimension: {traj_3d.dimension()}")
# Dimension: 3

# Create 12D trajectory (custom)
traj_12d = bh.DTrajectory(12)
print(f"Dimension: {traj_12d.dimension()}")
# Dimension: 12
