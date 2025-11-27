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
    state_cart = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
    traj_cart.add(epoch, state_cart)

print(f"Original representation: {traj_cart.representation}")
# Output: OrbitRepresentation.CARTESIAN

# Convert to Keplerian with degrees
traj_kep = traj_cart.to_keplerian(bh.AngleFormat.DEGREES)

print(f"Converted representation: {traj_kep.representation}")
# Output: OrbitRepresentation.KEPLERIAN
print(f"Angle format: {traj_kep.angle_format}")
# Output: AngleFormat.DEGREES

# Examine Keplerian elements
for epoch, oe in traj_kep:
    print(f"\nEpoch: {epoch}")
    print(f"  Semi-major axis: {oe[0] / 1e3:.2f} km")
    print(f"  Eccentricity: {oe[1]:.6f}")
    print(f"  Inclination: {oe[2]:.2f}°")
    print(f"  RAAN: {oe[3]:.2f}°")
    print(f"  Argument of perigee: {oe[4]:.2f}°")
    print(f"  Mean anomaly: {oe[5]:.2f}°")

# Output:
# Original representation: Cartesian
# Converted representation: Keplerian
# Angle format: Degrees

# Epoch: 2024-01-01 00:00:00.000 UTC
#   Semi-major axis: 6878.14 km
#   Eccentricity: 0.001000
#   Inclination: 97.80°
#   RAAN: 15.00°
#   Argument of perigee: 30.00°
#   True anomaly: 0.00°
