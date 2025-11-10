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
# Orbital elements (degrees):
#   a    = 6878136.300 m = 500.0 km altitude
#   e    = 0.0100
#   i    = 97.8000°
#   Ω    = 15.0000°
#   ω    = 30.0000°
#   M    = 45.0000°

# Convert to EME2000 Cartesian state and extract position
state_eme2000 = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
pos_eme2000 = state_eme2000[0:3]

print("Position in EME2000:")
print(f"  [{pos_eme2000[0]:.3f}, {pos_eme2000[1]:.3f}, {pos_eme2000[2]:.3f}] m\n")
# Position in EME2000:
#   [1848964.106, -434937.468, 6560410.530] m

# Transform to GCRF (constant transformation, no epoch needed)
pos_gcrf = bh.position_eme2000_to_gcrf(pos_eme2000)

print("Position in GCRF:")
print(f"  [{pos_gcrf[0]:.3f}, {pos_gcrf[1]:.3f}, {pos_gcrf[2]:.3f}] m\n")
# Position in GCRF:
# [1848963.547, -434937.816, 6560410.665] m

# Verify using rotation matrix
R_eme2000_to_gcrf = bh.rotation_eme2000_to_gcrf()
pos_gcrf_matrix = R_eme2000_to_gcrf @ pos_eme2000

print("Position in GCRF (using rotation matrix):")
print(
    f"  [{pos_gcrf_matrix[0]:.3f}, {pos_gcrf_matrix[1]:.3f}, {pos_gcrf_matrix[2]:.3f}] m\n"
)
# Position in GCRF (using rotation matrix):
# [1848963.547, -434937.816, 6560410.665] m

# Verify both methods give same result
diff = np.linalg.norm(pos_gcrf - pos_gcrf_matrix)
print(f"Difference between methods: {diff:.6e} m")
print("\nNote: Transformation is constant (time-independent, no epoch needed)")
# Difference between methods: 0.000000e+00 m
