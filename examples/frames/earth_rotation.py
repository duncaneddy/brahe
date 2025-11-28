# /// script
# dependencies = ["brahe"]
# ///
"""
Get the Earth Rotation matrix
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define epoch
epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)

# Get Earth rotation matrix (CIRS to TIRS transformation)
R_er = bh.earth_rotation(epc)

print(f"Epoch: {epc.to_datetime()}")
print("\nEarth Rotation matrix:")
print("Transforms from CIRS to TIRS")
print(f"  [{R_er[0, 0]:10.7f}, {R_er[0, 1]:10.7f}, {R_er[0, 2]:10.7f}]")
print(f"  [{R_er[1, 0]:10.7f}, {R_er[1, 1]:10.7f}, {R_er[1, 2]:10.7f}]")
print(f"  [{R_er[2, 0]:10.7f}, {R_er[2, 1]:10.7f}, {R_er[2, 2]:10.7f}]\n")
# [ 0.1794542, -0.9837663,  0.0000000]
# [ 0.9837663,  0.1794542,  0.0000000]
# [ 0.0000000,  0.0000000,  1.0000000]

# Define orbital elements in degrees
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

# Convert to GCRF and then to CIRS
state_gcrf = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
pos_gcrf = state_gcrf[0:3]
R_bpn = bh.bias_precession_nutation(epc)
pos_cirs = R_bpn @ pos_gcrf

print("Satellite position in CIRS:")
print(f"  [{pos_cirs[0]:.3f}, {pos_cirs[1]:.3f}, {pos_cirs[2]:.3f}] m\n")
# [1833728.342, -435153.781, 6564671.107] m

# Apply Earth rotation to get TIRS
pos_tirs = R_er @ pos_cirs

print("Satellite position in TIRS:")
print(f"  [{pos_tirs[0]:.3f}, {pos_tirs[1]:.3f}, {pos_tirs[2]:.3f}] m")
# [757159.942, 1725870.003, 6564671.107] m

# Calculate the magnitude of the change
diff = np.linalg.norm(pos_cirs - pos_tirs)
print(f"\nPosition change magnitude: {diff:.3f} m")
print("Note: Earth rotation causes large position changes (km scale)")
print(f"      due to ~{np.degrees(bh.OMEGA_EARTH * 3600):.3f}° rotation per hour")
# Position change magnitude: 2414337.034 m
