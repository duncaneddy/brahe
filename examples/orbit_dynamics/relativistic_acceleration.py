# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute general relativistic correction to satellite acceleration
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define GPS satellite state (MEO orbit where relativity is measurable)
a = bh.R_EARTH + 20180e3  # Semi-major axis (m)
e = 0.01  # Eccentricity
i = np.radians(55.0)  # Inclination (rad)
raan = np.radians(30.0)  # RAAN (rad)
argp = np.radians(45.0)  # Argument of perigee (rad)
nu = np.radians(90.0)  # True anomaly (rad)

# Convert to Cartesian state
oe = np.array([a, e, i, raan, argp, nu])
state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

print("GPS Satellite state (ECI):")
print(
    f"  Position: [{state[0] / 1e3:.1f}, {state[1] / 1e3:.1f}, {state[2] / 1e3:.1f}] km"
)
print(
    f"  Velocity: [{state[3] / 1e3:.3f}, {state[4] / 1e3:.3f}, {state[5] / 1e3:.3f}] km/s"
)
r_mag = np.linalg.norm(state[0:3])
v_mag = np.linalg.norm(state[3:6])
print(f"  Altitude: {(r_mag - bh.R_EARTH) / 1e3:.1f} km")
print(f"  Speed: {v_mag / 1e3:.3f} km/s")

# Compute relativistic acceleration
accel_rel = bh.accel_relativity(state)

print("\nRelativistic acceleration (m/s²):")
print(f"  ax = {accel_rel[0]:.15f}")
print(f"  ay = {accel_rel[1]:.15f}")
print(f"  az = {accel_rel[2]:.15f}")
print(f"  Magnitude: {np.linalg.norm(accel_rel):.15e} m/s²")

# Compare to Newtonian point-mass gravity
accel_newton = bh.accel_point_mass_gravity(
    state[0:3], np.array([0.0, 0.0, 0.0]), bh.GM_EARTH
)
accel_newton_mag = np.linalg.norm(accel_newton)

print(f"\nNewtonian gravity magnitude: {accel_newton_mag:.9f} m/s²")
print(
    f"Relativistic/Newtonian ratio: {np.linalg.norm(accel_rel) / accel_newton_mag:.6e}"
)

# Estimate accumulated position error if relativity is ignored
# Using simple approximation: Δr ≈ 0.5 * a * t²
# For 1 day propagation
one_day = 86400.0  # seconds
pos_error_1day = 0.5 * np.linalg.norm(accel_rel) * one_day**2

print("\nApproximate position error if relativity ignored:")
print(f"  After 1 day: {pos_error_1day:.3f} m")
print(f"  After 1 week: {pos_error_1day * 7:.1f} m")

# Compare to other perturbations at this altitude
# J2 magnitude (approximate)
j2 = 1.08263e-3
accel_j2_approx = 1.5 * j2 * bh.GM_EARTH * (bh.R_EARTH / r_mag) ** 2 / r_mag**2

# Third-body (Sun, approximate)
accel_sun_approx = 5e-8  # Typical value for GPS altitude

print("\nRelative magnitude of perturbations at GPS altitude:")
print(f"  J2: ~{accel_j2_approx:.6e} m/s²")
print(f"  Sun: ~{accel_sun_approx:.6e} m/s²")
print(f"  Relativity: {np.linalg.norm(accel_rel):.6e} m/s²")
print(f"  Relativity/J2 ratio: {np.linalg.norm(accel_rel) / accel_j2_approx:.6e}")

# Expected output:
# GPS Satellite state (ECI):
#   Position: [-21864.6, -435.7, 15074.0] km
#   Velocity: [-1.555, -2.730, -2.266] km/s
#   Altitude: 20182.7 km
#   Speed: 3.874 km/s

# Relativistic acceleration (m/s²):
#   ax = -0.000000000234510
#   ay = -0.000000000007302
#   az = 0.000000000158426
#   Magnitude: 2.831022208577214e-10 m/s²

# Newtonian gravity magnitude: 0.565009481 m/s²
# Relativistic/Newtonian ratio: 5.010575e-10

# Approximate position error if relativity ignored:
#   After 1 day: 1.057 m
#   After 1 week: 7.4 m

# Relative magnitude of perturbations at GPS altitude:
#   J2: ~5.290937e-05 m/s²
#   Sun: ~5.000000e-08 m/s²
#   Relativity: 2.831022e-10 m/s²
#   Relativity/J2 ratio: 5.350701e-06
