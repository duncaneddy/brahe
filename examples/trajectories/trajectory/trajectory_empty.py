# /// script
# dependencies = ["brahe"]
# ///
"""
Create empty Trajectory instances with different dimensions
"""

import brahe as bh

bh.initialize_eop()

# Create 6D trajectory (default)
traj = bh.Trajectory()
print(f"Dimension: {traj.dimension()}")
# Dimension: 6

# Create 3D trajectory (position only)
traj_3d = bh.Trajectory(3)
print(f"Dimension: {traj_3d.dimension()}")
# Dimension: 3

# Create 12D trajectory (custom)
traj_12d = bh.Trajectory(12)
print(f"Dimension: {traj_12d.dimension()}")
# Dimension: 12
