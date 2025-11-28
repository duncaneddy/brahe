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
    bh.OrbitFrame.ECI,
    bh.OrbitRepresentation.CARTESIAN,
    None,  # No angle format for Cartesian
)
print(f"Frame (str): {traj_eci.frame}")  # Output: ECI
print(
    f"Frame (repr): {repr(traj_eci.frame)}"
)  # Output: OrbitFrame(Earth-Centered Inertial)
print(f"Representation (str): {traj_eci.representation}")  # Output: Cartesian
print(
    f"Representation (repr): {repr(traj_eci.representation)}"
)  # Output: OrbitRepresentation(Cartesian)

# Create trajectory in ECEF frame, Cartesian representation
traj_ecef = bh.OrbitTrajectory(
    6, bh.OrbitFrame.ECEF, bh.OrbitRepresentation.CARTESIAN, None
)
print(f"Frame (str): {traj_ecef.frame}")  # Output: ECEF
print(
    f"Frame (repr): {repr(traj_ecef.frame)}"
)  # Output: OrbitFrame(Earth-Centered Earth-Fixed)

# Output:
# Frame (str): ECI
# Frame (repr): OrbitFrame(Earth-Centered Inertial)
# Representation (str): Cartesian
# Representation (repr): OrbitRepresentation(Cartesian)
# Frame (str): ECEF
# Frame (repr): OrbitFrame(Earth-Centered Earth-Fixed)
