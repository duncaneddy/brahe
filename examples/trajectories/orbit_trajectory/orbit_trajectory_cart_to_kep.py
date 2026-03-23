# /// script
# dependencies = ["brahe"]
# ///
"""
Convert trajectory from Cartesian to Keplerian representation
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory in ECI Cartesian
traj_cart = bh.OrbitTrajectory(
    6, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None
)

# Add Cartesian states
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
for i in range(3):
    epoch = epoch0 + i * 300.0  # 5-minute intervals
    # Use orbital elements to create realistic Cartesian states
    oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, i * 10.0])
    state_cart = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    traj_cart.add(epoch, state_cart)

print(f"Original representation: {traj_cart.representation}")

# Convert to Keplerian with degrees
traj_kep = traj_cart.to_keplerian(bh.AngleFormat.DEGREES)

print(f"Converted representation: {traj_kep.representation}")
print(f"Angle format: {traj_kep.angle_format}")

# Examine Keplerian elements
for epoch, oe in traj_kep:
    print(f"\nEpoch: {epoch}")
    print(f"  Semi-major axis: {oe[0] / 1e3:.2f} km")
    print(f"  Eccentricity: {oe[1]:.6f}")
    print(f"  Inclination: {oe[2]:.2f}°")
    print(f"  RAAN: {oe[3]:.2f}°")
    print(f"  Argument of perigee: {oe[4]:.2f}°")
    print(f"  Mean anomaly: {oe[5]:.2f}°")
