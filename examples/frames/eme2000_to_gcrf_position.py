# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Transform position vector from EME2000 to GCRF
"""

import brahe as bh
import numpy as np

# Define orbital elements in degrees
# LEO satellite: 500 km altitude, sun-synchronous orbit
oe = np.array(
    [
        bh.R_EARTH + 500e3,  # Semi-major axis (m)
        0.01,  # Eccentricity
        97.8,  # Inclination (deg)
        15.0,  # Right ascension of ascending node (deg)
        30.0,  # Argument of periapsis (deg)
        45.0,  # Mean anomaly (deg)
    ]
)

print("Orbital elements (degrees):")
print(f"  a    = {oe[0]:.3f} m = {(oe[0] - bh.R_EARTH) / 1e3:.1f} km altitude")
print(f"  e    = {oe[1]:.4f}")
print(f"  i    = {oe[2]:.4f}°")
print(f"  Ω    = {oe[3]:.4f}°")
print(f"  ω    = {oe[4]:.4f}°")
print(f"  M    = {oe[5]:.4f}°\n")

# Convert to EME2000 Cartesian state and extract position
state_eme2000 = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
pos_eme2000 = state_eme2000[0:3]

print("Position in EME2000:")
print(f"  [{pos_eme2000[0]:.3f}, {pos_eme2000[1]:.3f}, {pos_eme2000[2]:.3f}] m\n")

# Transform to GCRF (constant transformation, no epoch needed)
pos_gcrf = bh.position_eme2000_to_gcrf(pos_eme2000)

print("Position in GCRF:")
print(f"  [{pos_gcrf[0]:.3f}, {pos_gcrf[1]:.3f}, {pos_gcrf[2]:.3f}] m\n")

R_eme2000_to_gcrf = bh.rotation_eme2000_to_gcrf()
pos_gcrf_matrix = R_eme2000_to_gcrf @ pos_eme2000

print("Position in GCRF (using rotation matrix):")
print(
    f"  [{pos_gcrf_matrix[0]:.3f}, {pos_gcrf_matrix[1]:.3f}, {pos_gcrf_matrix[2]:.3f}] m\n"
)

diff = np.linalg.norm(pos_gcrf - pos_gcrf_matrix)
print(f"Difference between methods: {diff:.6e} m")
print("\nNote: Transformation is constant (time-independent, no epoch needed)")
