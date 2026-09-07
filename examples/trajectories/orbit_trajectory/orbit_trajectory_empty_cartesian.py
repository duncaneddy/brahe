# /// script
# dependencies = ["brahe"]
# ///
"""
Create empty OrbitTrajectory in Cartesian representation
"""

import brahe as bh

bh.initialize_eop()

# Create trajectory in ECI frame, Cartesian representation
traj_eci = bh.OrbitTrajectory(
    6,  # State dimension (position + velocity)
    bh.CelestialFrame.ECI,
    bh.OrbitRepresentation.CARTESIAN,
    None,  # No angle format for Cartesian
)
print(f"Frame (str): {traj_eci.frame}")  # Output: GCRF
print(f"Frame (repr): {traj_eci.frame!r}")  # Output: ReferenceFrame("GCRF")
print(f"Representation (str): {traj_eci.representation}")  # Output: Cartesian
print(
    f"Representation (repr): {traj_eci.representation!r}"
)  # Output: OrbitRepresentation(Cartesian)

# Create trajectory in ECEF frame, Cartesian representation
traj_ecef = bh.OrbitTrajectory(
    6, bh.CelestialFrame.ECEF, bh.OrbitRepresentation.CARTESIAN, None
)
print(f"Frame (str): {traj_ecef.frame}")  # Output: ITRF
print(f"Frame (repr): {traj_ecef.frame!r}")  # Output: ReferenceFrame("ITRF")
