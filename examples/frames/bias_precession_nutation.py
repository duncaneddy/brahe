# /// script
# dependencies = ["brahe"]
# ///
"""
Get the Bias-Precession-Nutation (BPN) rotation matrix
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define epoch
epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)

# Get BPN matrix (GCRF to CIRS transformation)
R_bpn = bh.bias_precession_nutation(epc)

print(f"Epoch: {epc.to_datetime()}")
print("\nBias-Precession-Nutation (BPN) matrix:")
print("Transforms from GCRF to CIRS")
print(f"  [{R_bpn[0, 0]:10.7f}, {R_bpn[0, 1]:10.7f}, {R_bpn[0, 2]:10.7f}]")
print(f"  [{R_bpn[1, 0]:10.7f}, {R_bpn[1, 1]:10.7f}, {R_bpn[1, 2]:10.7f}]")
print(f"  [{R_bpn[2, 0]:10.7f}, {R_bpn[2, 1]:10.7f}, {R_bpn[2, 2]:10.7f}]\n")

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

# Convert to GCRF (ECI) position
state_gcrf = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
pos_gcrf = state_gcrf[0:3]

print("Satellite position in GCRF:")
print(f"  [{pos_gcrf[0]:.3f}, {pos_gcrf[1]:.3f}, {pos_gcrf[2]:.3f}] m\n")

# Transform to CIRS using BPN matrix
pos_cirs = R_bpn @ pos_gcrf

print("Satellite position in CIRS:")
print(f"  [{pos_cirs[0]:.3f}, {pos_cirs[1]:.3f}, {pos_cirs[2]:.3f}] m")

# Calculate the magnitude of the change
diff = np.linalg.norm(pos_gcrf - pos_cirs)
print(f"\nPosition change magnitude: {diff:.3f} m")
print("Note: BPN effects are typically meters to tens of meters")

# Expected output:
# Position in CIRS: [3476690.567, 5527916.087, 2649962.165] m
# Position change magnitude: ~39 m
