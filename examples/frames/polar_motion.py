# /// script
# dependencies = ["brahe"]
# ///
"""
Get the Polar Motion matrix
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define epoch
epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)

# Get polar motion matrix (TIRS to ITRF transformation)
R_pm = bh.polar_motion(epc)

print(f"Epoch: {epc.to_datetime()}")
print("\nPolar Motion matrix:")
print("Transforms from TIRS to ITRF")
print(f"  [{R_pm[0, 0]:10.7f}, {R_pm[0, 1]:10.7f}, {R_pm[0, 2]:10.7f}]")
print(f"  [{R_pm[1, 0]:10.7f}, {R_pm[1, 1]:10.7f}, {R_pm[1, 2]:10.7f}]")
print(f"  [{R_pm[2, 0]:10.7f}, {R_pm[2, 1]:10.7f}, {R_pm[2, 2]:10.7f}]\n")

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

# Convert through the full chain: GCRF → CIRS → TIRS
state_gcrf = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
pos_gcrf = state_gcrf[0:3]
R_bpn = bh.bias_precession_nutation(epc)
R_er = bh.earth_rotation(epc)
pos_tirs = R_er @ R_bpn @ pos_gcrf

print("Satellite position in TIRS:")
print(f"  [{pos_tirs[0]:.3f}, {pos_tirs[1]:.3f}, {pos_tirs[2]:.3f}] m\n")

# Apply polar motion to get ITRF (ECEF)
pos_itrf = R_pm @ pos_tirs

print("Satellite position in ITRF (ECEF):")
print(f"  [{pos_itrf[0]:.3f}, {pos_itrf[1]:.3f}, {pos_itrf[2]:.3f}] m")

# Calculate the magnitude of the change
diff = np.linalg.norm(pos_tirs - pos_itrf)
print(f"\nPosition change magnitude: {diff:.3f} m")
print("Note: Polar motion effects are typically centimeters to meters")

# Verify against full transformation
pos_ecef_direct = bh.position_eci_to_ecef(epc, pos_gcrf)
print("\nVerification using position_eci_to_ecef:")
print(
    f"  [{pos_ecef_direct[0]:.3f}, {pos_ecef_direct[1]:.3f}, {pos_ecef_direct[2]:.3f}] m"
)
print(f"  Max difference: {np.max(np.abs(pos_itrf - pos_ecef_direct)):.2e} m")

# Expected output:
# Position in ITRF: [3210319.128, 5246384.459, 2649959.679] m
# Position change magnitude: ~0.3 m
# Max difference: ~1e-9 m (numerical precision)
