# /// script
# dependencies = ["brahe"]
# ///
"""
Get ITRF to GCRF rotation matrix and use it to transform position vectors
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define epoch
epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)

# Get rotation matrix from ITRF to GCRF
R_itrf_to_gcrf = bh.rotation_itrf_to_gcrf(epc)

print(f"Epoch: {epc.to_datetime()}")
print("\nITRF to GCRF rotation matrix:")
print(
    f"  [{R_itrf_to_gcrf[0, 0]:10.7f}, {R_itrf_to_gcrf[0, 1]:10.7f}, {R_itrf_to_gcrf[0, 2]:10.7f}]"
)
print(
    f"  [{R_itrf_to_gcrf[1, 0]:10.7f}, {R_itrf_to_gcrf[1, 1]:10.7f}, {R_itrf_to_gcrf[1, 2]:10.7f}]"
)
print(
    f"  [{R_itrf_to_gcrf[2, 0]:10.7f}, {R_itrf_to_gcrf[2, 1]:10.7f}, {R_itrf_to_gcrf[2, 2]:10.7f}]\n"
)
# [ 0.1794538,  0.9837637,  0.0023225]
# [-0.9837663,  0.1794542,  0.0000338]
# [-0.0003836, -0.0022908,  0.9999973]

# Verify it's the transpose of GCRF to ITRF rotation
R_gcrf_to_itrf = bh.rotation_gcrf_to_itrf(epc)
print("Verification: R_itrf_to_gcrf = R_gcrf_to_itrf^T")
print(f"  Max difference: {np.max(np.abs(R_itrf_to_gcrf - R_gcrf_to_itrf.T)):.2e}\n")
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

# Convert to GCRF Cartesian state and extract position
state_gcrf = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Transform to ITRF
pos_itrf = bh.position_gcrf_to_itrf(epc, state_gcrf[0:3])

print("Satellite position in ITRF:")
print(f"  [{pos_itrf[0]:.3f}, {pos_itrf[1]:.3f}, {pos_itrf[2]:.3f}] m\n")
# [757164.267, 1725863.563, 6564672.302] m

# Transform back to GCRF using rotation matrix
pos_gcrf = R_itrf_to_gcrf @ pos_itrf

print("Satellite position in GCRF (using rotation matrix):")
print(f"  [{pos_gcrf[0]:.3f}, {pos_gcrf[1]:.3f}, {pos_gcrf[2]:.3f}] m")
# [1848964.106, -434937.468, 6560410.530] m

# Verify using position transformation function
pos_gcrf_direct = bh.position_itrf_to_gcrf(epc, pos_itrf)
print("\nSatellite position in GCRF (using position_itrf_to_gcrf):")
print(
    f"  [{pos_gcrf_direct[0]:.3f}, {pos_gcrf_direct[1]:.3f}, {pos_gcrf_direct[2]:.3f}] m"
)
# [1848964.106, -434937.468, 6560410.530] m
