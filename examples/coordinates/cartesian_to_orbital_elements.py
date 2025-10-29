# /// script
# dependencies = ["brahe"]
# ///
"""
Convert between Keplerian orbital elements and Cartesian state vectors
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define Cartesian state vector [px, py, pz, vx, vy, vz] in meters and meters per second
state = np.array(
    [1848964.106, -434937.468, 6560410.530, -7098.379734, -2173.344867, 1913.333385]
)

# Convert orbital elements to Cartesian state using degrees
oe_deg = bh.state_cartesian_to_osculating(state, bh.AngleFormat.DEGREES)
print("Osculating state [a, e, i, Ω, ω, M] (deg):")
print(f"Semi-major axis (m): {oe_deg[0]:.3f}")
print(f"Eccentricity: {oe_deg[1]:.6f}")
print(f"Inclination (deg): {oe_deg[2]:.6f}")
print(f"RA of ascending node (deg): {oe_deg[3]:.6f}")
print(f"Argument of periapsis (deg): {oe_deg[4]:.6f}")
print(f"Mean anomaly (deg): {oe_deg[5]:.6f}")
# Osculating state  (deg):
# Semi-major axis (m): 6878136.299
# Eccentricity: 0.010000
# Inclination (deg): 97.800000
# RA of ascending node (deg): 15.000000
# Argument of periapsis (deg): 30.000000
# Mean anomaly (deg): 45.000000

# You can also convert using radians
oe_rad = bh.state_cartesian_to_osculating(state, bh.AngleFormat.RADIANS)
print("\nOsculating state [a, e, i, Ω, ω, M] (rad):")
print(f"Semi-major axis (m): {oe_rad[0]:.3f}")
print(f"Eccentricity: {oe_rad[1]:.6f}")
print(f"Inclination (rad): {oe_rad[2]:.6f}")
print(f"RA of ascending node (rad): {oe_rad[3]:.6f}")
print(f"Argument of periapsis (rad): {oe_rad[4]:.6f}")
print(f"Mean anomaly (rad): {oe_rad[5]:.6f}")
# Osculating state  (rad):
# Semi-major axis (m): 6878136.299
# Eccentricity: 0.010000
# Inclination (rad): 1.706932
# RA of ascending node (rad): 0.261799
# Argument of periapsis (rad): 0.523599
# Mean anomaly (rad): 0.785398
