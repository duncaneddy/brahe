# /// script
# dependencies = ["brahe"]
# ///
"""
Convert to Keplerian with different angle formats (radians vs degrees)
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory in ECI Cartesian
traj_cart = bh.SOrbitTrajectory(
    bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None
)

# Add a state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, 0.0])
state_cart = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
traj_cart.add(epoch, state_cart)

# Convert to Keplerian with radians
traj_kep_rad = traj_cart.to_keplerian(bh.AngleFormat.RADIANS)
_, oe_rad = traj_kep_rad.first()

# Convert to Keplerian with degrees
traj_kep_deg = traj_cart.to_keplerian(bh.AngleFormat.DEGREES)
_, oe_deg = traj_kep_deg.first()

print("Radians version:")
print(f"  Inclination: {oe_rad[2]:.6f} rad = {np.degrees(oe_rad[2]):.2f}°")

print("\nDegrees version:")
print(f"  Inclination: {oe_deg[2]:.2f}°")

# Output:
# Radians version:
#   Inclination: 0.900000 rad = 51.57°

# Degrees version:
#   Inclination: 51.57°
