# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute point-mass gravitational acceleration for an Earth satellite
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define satellite position in ECI frame (LEO satellite at 500 km altitude)
# Using Keplerian elements and converting to Cartesian
a = bh.R_EARTH + 500e3  # Semi-major axis (m)
e = 0.001  # Eccentricity
i = 97.8  # Inclination (deg)
raan = 0.0  # RAAN (deg)
argp = 0.0  # Argument of perigee (deg)
nu = 0.0  # True anomaly (deg)

# Convert to Cartesian state
oe = np.array([a, e, i, raan, argp, nu])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
r_sat = state[0:3]  # Position vector (m)

print("Satellite position (ECI, m):")
print(f"  x = {r_sat[0]:.3f}")
print(f"  y = {r_sat[1]:.3f}")
print(f"  z = {r_sat[2]:.3f}")

# Compute point-mass gravitational acceleration
# For Earth-centered case, central body is at origin
r_earth = np.array([0.0, 0.0, 0.0])
accel = bh.accel_point_mass_gravity(r_sat, r_earth, bh.GM_EARTH)

print("\nPoint-mass gravity acceleration (m/s²):")
print(f"  ax = {accel[0]:.6f}")
print(f"  ay = {accel[1]:.6f}")
print(f"  az = {accel[2]:.6f}")

# Compute magnitude
accel_mag = np.linalg.norm(accel)
print(f"\nAcceleration magnitude: {accel_mag:.6f} m/s²")

# Compare to theoretical value: GM/r²
r_mag = np.linalg.norm(r_sat)
accel_theoretical = bh.GM_EARTH / (r_mag**2)
print(f"Theoretical magnitude: {accel_theoretical:.6f} m/s²")

# Expected output:
# Satellite position (ECI, m):
#   x = 6871258.164
#   y = 0.000
#   z = 0.000

# Point-mass gravity acceleration (m/s²):
#   ax = -8.442387
#   ay = -0.000000
#   az = -0.000000

# Acceleration magnitude: 8.442387 m/s²
# Theoretical magnitude: 8.442387 m/s²
