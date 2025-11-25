# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute spherical harmonic gravitational acceleration for an Earth satellite
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create an epoch for frame transformations
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Define satellite position in ECI frame (LEO satellite at 500 km altitude)
a = bh.R_EARTH + 500e3  # Semi-major axis (m)
e = 0.001  # Eccentricity
i = 97.8  # Inclination (deg)
raan = 45.0  # RAAN (deg)
argp = 30.0  # Argument of perigee (deg)
nu = 60.0  # True anomaly (deg)

# Convert to Cartesian state
oe = np.array([a, e, i, raan, argp, nu])
state_eci = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
r_eci = state_eci[0:3]  # Position vector (m)

print("Satellite position (ECI, m):")
print(f"  x = {r_eci[0]:.3f}")
print(f"  y = {r_eci[1]:.3f}")
print(f"  z = {r_eci[2]:.3f}")

# Load gravity model (GGM05S - degree/order 180)
gravity_model = bh.GravityModel.from_model_type(bh.GravityModelType.GGM05S)
print(
    f"\nGravity model: GGM05S (max degree {gravity_model.n_max}, max order {gravity_model.m_max})"
)

# For spherical harmonics, we need the ECI to body-fixed rotation matrix
# This rotates from ECI (inertial) to ECEF (Earth-fixed) frame
R_eci_ecef = bh.rotation_eci_to_ecef(epoch)

# Compute spherical harmonic acceleration (degree 10, order 10)
n_max = 10
m_max = 10
accel_sh = bh.accel_gravity_spherical_harmonics(
    r_eci, R_eci_ecef, gravity_model, n_max, m_max
)

print(f"\nSpherical harmonic acceleration (degree {n_max}, order {m_max}):")
print(f"  ax = {accel_sh[0]:.9f} m/s²")
print(f"  ay = {accel_sh[1]:.9f} m/s²")
print(f"  az = {accel_sh[2]:.9f} m/s²")

# Compute point-mass for comparison
accel_pm = bh.accel_point_mass_gravity(r_eci, np.array([0.0, 0.0, 0.0]), bh.GM_EARTH)

print("\nPoint-mass acceleration:")
print(f"  ax = {accel_pm[0]:.9f} m/s²")
print(f"  ay = {accel_pm[1]:.9f} m/s²")
print(f"  az = {accel_pm[2]:.9f} m/s²")

# Compute difference (perturbation due to non-spherical Earth)
accel_pert = accel_sh - accel_pm

print("\nPerturbation (spherical harmonics - point mass):")
print(f"  Δax = {accel_pert[0]:.9f} m/s²")
print(f"  Δay = {accel_pert[1]:.9f} m/s²")
print(f"  Δaz = {accel_pert[2]:.9f} m/s²")
print(f"  Magnitude: {np.linalg.norm(accel_pert):.9f} m/s²")

# Expected output:
# Satellite position (ECI, m):
#   x = 651307.572
#   y = -668157.599
#   z = 6811086.322

# Gravity model: GGM05S (max degree 180, max order 180)

# Spherical harmonic acceleration (degree 10, order 10):
#   ax = -0.794811805 m/s²
#   ay = 0.815141691 m/s²
#   az = -8.333760910 m/s²

# Point-mass acceleration:
#   ax = -0.799028363 m/s²
#   ay = 0.819700085 m/s²
#   az = -8.355884974 m/s²

# Perturbation (spherical harmonics - point mass):
#   Δax = 0.004216558 m/s²
#   Δay = -0.004558395 m/s²
#   Δaz = 0.022124064 m/s²
#   Magnitude: 0.022978958 m/s²
