# /// script
# dependencies = ["brahe"]
# ///
"""
Create empty SOrbitTrajectory in Keplerian representation
"""

import brahe as bh

bh.initialize_eop()

# Create trajectory in ECI frame, Keplerian representation with radians
traj_kep_rad = bh.SOrbitTrajectory(
    bh.OrbitFrame.ECI,
    bh.OrbitRepresentation.KEPLERIAN,
    bh.AngleFormat.RADIANS,  # Required for Keplerian
)

# Create trajectory in ECI frame, Keplerian representation with degrees
traj_kep_deg = bh.SOrbitTrajectory(
    bh.OrbitFrame.ECI, bh.OrbitRepresentation.KEPLERIAN, bh.AngleFormat.DEGREES
)
