# /// script
# dependencies = ["brahe"]
# ///
"""
Get ECI to ECEF rotation matrix and use it to transform position vectors
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define epoch
epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)

# Get rotation matrix from ECI to ECEF
R_eci_to_ecef = bh.rotation_eci_to_ecef(epc)

print(f"Epoch: {epc}")  # Epoch: 2024-01-01 12:00:00 UTC
print("\nECI to ECEF rotation matrix:")
print(
    f"  [{R_eci_to_ecef[0, 0]:10.7f}, {R_eci_to_ecef[0, 1]:10.7f}, {R_eci_to_ecef[0, 2]:10.7f}]"
)
print(
    f"  [{R_eci_to_ecef[1, 0]:10.7f}, {R_eci_to_ecef[1, 1]:10.7f}, {R_eci_to_ecef[1, 2]:10.7f}]"
)
print(
    f"  [{R_eci_to_ecef[2, 0]:10.7f}, {R_eci_to_ecef[2, 1]:10.7f}, {R_eci_to_ecef[2, 2]:10.7f}]\n"
)
# [ 0.1794538, -0.9837663, -0.0003836]
# [ 0.9837637,  0.1794542, -0.0022908]
# [ 0.0023225,  0.0000338,  0.9999973]

# Define orbital elements in degrees for satellite position
oe = np.array(
    [
        bh.R_EARTH + 500e3,  # Semi-major axis (m)
        0.01,  # Eccentricity
        97.8,  # Inclination (deg)
        15.0,  # RAAN (deg)
        30.0,  # Argument of periapsis (deg)
        45.0,  # Mean anomaly (deg)
    ]
)

# Convert to ECI Cartesian state and extract position
state_eci = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
pos_eci = state_eci[0:3]

print("Position in ECI:")
print(f"  [{pos_eci[0]:.3f}, {pos_eci[1]:.3f}, {pos_eci[2]:.3f}] m\n")
# [1848964.106, -434937.468, 6560410.530] m

# Transform position using rotation matrix
pos_ecef = R_eci_to_ecef @ pos_eci

print("Position in ECEF (using rotation matrix):")
print(f"  [{pos_ecef[0]:.3f}, {pos_ecef[1]:.3f}, {pos_ecef[2]:.3f}] m")
# [757164.267, 1725863.563, 6564672.302] m

# Verify using position transformation function
pos_ecef_direct = bh.position_eci_to_ecef(epc, pos_eci)
print("\nPosition in ECEF (using position_eci_to_ecef):")
print(
    f"  [{pos_ecef_direct[0]:.3f}, {pos_ecef_direct[1]:.3f}, {pos_ecef_direct[2]:.3f}] m"
)
# [757164.267, 1725863.563, 6564672.302] m
