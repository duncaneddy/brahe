# /// script
# dependencies = ["brahe"]
# ///
"""
Get ECEF to ECI rotation matrix and use it to transform position vectors
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define epoch
epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)

# Get rotation matrix from ECEF to ECI
R_ecef_to_eci = bh.rotation_ecef_to_eci(epc)

print(f"Epoch: {epc.to_datetime()}")
print("\nECEF to ECI rotation matrix:")
print(
    f"  [{R_ecef_to_eci[0, 0]:10.7f}, {R_ecef_to_eci[0, 1]:10.7f}, {R_ecef_to_eci[0, 2]:10.7f}]"
)
print(
    f"  [{R_ecef_to_eci[1, 0]:10.7f}, {R_ecef_to_eci[1, 1]:10.7f}, {R_ecef_to_eci[1, 2]:10.7f}]"
)
print(
    f"  [{R_ecef_to_eci[2, 0]:10.7f}, {R_ecef_to_eci[2, 1]:10.7f}, {R_ecef_to_eci[2, 2]:10.7f}]\n"
)
# [ 0.1794538,  0.9837637,  0.0023225]
# [-0.9837663,  0.1794542,  0.0000338]
# [-0.0003836, -0.0022908,  0.9999973]

# Verify it's the transpose of ECI to ECEF rotation
R_eci_to_ecef = bh.rotation_eci_to_ecef(epc)
print("Verification: R_ecef_to_eci = R_eci_to_ecef^T")
print(f"  Max difference: {np.max(np.abs(R_ecef_to_eci - R_eci_to_ecef.T)):.2e}\n")
# Max difference: 0.00e0

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

# Transform to ECEF
pos_ecef = bh.position_eci_to_ecef(epc, state_eci[0:3])

print("Satellite position in ECEF:")
print(f"  [{pos_ecef[0]:.3f}, {pos_ecef[1]:.3f}, {pos_ecef[2]:.3f}] m\n")
# [757164.267, 1725863.563, 6564672.302] m

# Transform back to ECI using rotation matrix
pos_eci = R_ecef_to_eci @ pos_ecef

print("Satellite position in ECI (using rotation matrix):")
print(f"  [{pos_eci[0]:.3f}, {pos_eci[1]:.3f}, {pos_eci[2]:.3f}] m")
# [1848964.106, -434937.468, 6560410.530] m

# Verify using position transformation function
pos_eci_direct = bh.position_ecef_to_eci(epc, pos_ecef)
print("\nSatellite position in ECI (using position_ecef_to_eci):")
print(
    f"  [{pos_eci_direct[0]:.3f}, {pos_eci_direct[1]:.3f}, {pos_eci_direct[2]:.3f}] m"
)
# [1848964.106, -434937.468, 6560410.530] m
