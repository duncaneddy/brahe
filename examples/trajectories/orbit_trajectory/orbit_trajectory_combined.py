# /// script
# dependencies = ["brahe"]
# ///
"""
Chain multiple frame and representation conversions
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Start with ECI Cartesian trajectory
traj_eci_cart = bh.SOrbitTrajectory(
    bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None
)

# Add states
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, 0.0])
state_cart = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
traj_eci_cart.add(epoch, state_cart)

print("Original:")
print(f"  Frame: {traj_eci_cart.frame}")
print(f"  Representation: {traj_eci_cart.representation}")

# Convert to ECEF frame (stays Cartesian)
traj_ecef_cart = traj_eci_cart.to_ecef()
print("\nAfter to_ecef():")
print(f"  Frame: {traj_ecef_cart.frame}")
print(f"  Representation: {traj_ecef_cart.representation}")

# Convert back to ECI
traj_eci_cart2 = traj_ecef_cart.to_eci()
print("\nAfter to_eci():")
print(f"  Frame: {traj_eci_cart2.frame}")
print(f"  Representation: {traj_eci_cart2.representation}")

# Convert to Keplerian (in ECI frame)
traj_eci_kep = traj_eci_cart2.to_keplerian(bh.AngleFormat.DEGREES)
print("\nAfter to_keplerian():")
print(f"  Frame: {traj_eci_kep.frame}")
print(f"  Representation: {traj_eci_kep.representation}")
print(f"  Angle format: {traj_eci_kep.angle_format}")
