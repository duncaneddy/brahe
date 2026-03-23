# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Transform state vector (position and velocity) from GCRF to EME2000
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

# Convert to EME2000 state, then transform to GCRF
# (Starting in EME2000 to get GCRF representation)
state_eme2000_orig = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
state_gcrf = bh.state_eme2000_to_gcrf(state_eme2000_orig)

print("GCRF state vector:")
print(f"  Position: [{state_gcrf[0]:.3f}, {state_gcrf[1]:.3f}, {state_gcrf[2]:.3f}] m")
print(
    f"  Velocity: [{state_gcrf[3]:.6f}, {state_gcrf[4]:.6f}, {state_gcrf[5]:.6f}] m/s\n"
)

# Transform to EME2000 (constant transformation, no epoch needed)
state_eme2000 = bh.state_gcrf_to_eme2000(state_gcrf)

print("EME2000 state vector:")
print(
    f"  Position: [{state_eme2000[0]:.3f}, {state_eme2000[1]:.3f}, {state_eme2000[2]:.3f}] m"
)
print(
    f"  Velocity: [{state_eme2000[3]:.6f}, {state_eme2000[4]:.6f}, {state_eme2000[5]:.6f}] m/s\n"
)

# Transform back to GCRF to verify round-trip
state_gcrf_back = bh.state_eme2000_to_gcrf(state_eme2000)

print("GCRF state vector (transformed from EME2000):")
print(
    f"  Position: [{state_gcrf_back[0]:.3f}, {state_gcrf_back[1]:.3f}, {state_gcrf_back[2]:.3f}] m"
)
print(
    f"  Velocity: [{state_gcrf_back[3]:.6f}, {state_gcrf_back[4]:.6f}, {state_gcrf_back[5]:.6f}] m/s\n"
)

diff_pos = np.linalg.norm(state_gcrf[0:3] - state_gcrf_back[0:3])
diff_vel = np.linalg.norm(state_gcrf[3:6] - state_gcrf_back[3:6])
print("Round-trip error:")
print(f"  Position: {diff_pos:.6e} m")
print(f"  Velocity: {diff_vel:.6e} m/s")
print("\nNote: Transformation is constant (time-independent, no epoch needed)")
