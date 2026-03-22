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
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
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
