# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute third-body gravitational perturbations from Sun and Moon
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create an epoch
epoch = bh.Epoch.from_datetime(2024, 6, 21, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Define satellite position (GPS-like MEO satellite at ~20,000 km altitude)
a = bh.R_EARTH + 20180e3  # Semi-major axis (m)
e = 0.01  # Eccentricity
i = 55.0  # Inclination (deg)
raan = 120.0  # RAAN (deg)
argp = 45.0  # Argument of perigee (deg)
nu = 90.0  # True anomaly (deg)

# Convert to Cartesian state
oe = np.array([a, e, i, raan, argp, nu])
state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
r_sat = state[0:3]  # Position vector (m)

print("Satellite position (ECI, m):")
print(f"  x = {r_sat[0] / 1e3:.1f} km")
print(f"  y = {r_sat[1] / 1e3:.1f} km")
print(f"  z = {r_sat[2] / 1e3:.1f} km")
print(f"  Altitude: {(np.linalg.norm(r_sat) - bh.R_EARTH) / 1e3:.1f} km")

# Compute Sun perturbation using analytical model
accel_sun = bh.accel_third_body_sun(epoch, r_sat)

print("\nSun third-body acceleration (analytical):")
print(f"  ax = {accel_sun[0]:.12f} m/s²")
print(f"  ay = {accel_sun[1]:.12f} m/s²")
print(f"  az = {accel_sun[2]:.12f} m/s²")
print(f"  Magnitude: {np.linalg.norm(accel_sun):.12f} m/s²")

# Compute Moon perturbation using analytical model
accel_moon = bh.accel_third_body_moon(epoch, r_sat)

print("\nMoon third-body acceleration (analytical):")
print(f"  ax = {accel_moon[0]:.12f} m/s²")
print(f"  ay = {accel_moon[1]:.12f} m/s²")
print(f"  az = {accel_moon[2]:.12f} m/s²")
print(f"  Magnitude: {np.linalg.norm(accel_moon):.12f} m/s²")

# Compute combined Sun + Moon acceleration
accel_combined = accel_sun + accel_moon

print("\nCombined Sun + Moon acceleration:")
print(f"  ax = {accel_combined[0]:.12f} m/s²")
print(f"  ay = {accel_combined[1]:.12f} m/s²")
print(f"  az = {accel_combined[2]:.12f} m/s²")
print(f"  Magnitude: {np.linalg.norm(accel_combined):.12f} m/s²")

# Compare Sun vs Moon relative magnitude
ratio = np.linalg.norm(accel_sun) / np.linalg.norm(accel_moon)
print(f"\nSun/Moon acceleration ratio: {ratio:.3f}")

# Expected output:
# Satellite position (ECI, m):
#   x = 435.7 km
#   y = -21864.6 km
#   z = 15074.0 km
#   Altitude: 20182.7 km

# Sun third-body acceleration (analytical):
#   ax = -0.000000011195 m/s²
#   ay = -0.000000636482 m/s²
#   az = -0.000001202965 m/s²
#   Magnitude: 0.000001361014 m/s²

# Moon third-body acceleration (analytical):
#   ax = -0.000000403111 m/s²
#   ay = -0.000000652316 m/s²
#   az = -0.000002920304 m/s²
#   Magnitude: 0.000003019303 m/s²

# Combined Sun + Moon acceleration:
#   ax = -0.000000414306 m/s²
#   ay = -0.000001288798 m/s²
#   az = -0.000004123269 m/s²
#   Magnitude: 0.000004339815 m/s²

# Sun/Moon acceleration ratio: 0.451
