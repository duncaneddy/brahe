# /// script
# dependencies = ["brahe"]
# ///
"""
Create empty OrbitTrajectory in Keplerian representation
"""

import brahe as bh

bh.initialize_eop()

# Create trajectory in ECI frame, Keplerian representation with radians
traj_kep_rad = bh.OrbitTrajectory(
    6,  # State dimension (6 orbital elements)
    bh.OrbitFrame.ECI,
    bh.OrbitRepresentation.KEPLERIAN,
    bh.AngleFormat.RADIANS,  # Required for Keplerian
)

# Create trajectory in ECI frame, Keplerian representation with degrees
traj_kep_deg = bh.OrbitTrajectory(
    6, bh.OrbitFrame.ECI, bh.OrbitRepresentation.KEPLERIAN, bh.AngleFormat.DEGREES
)
