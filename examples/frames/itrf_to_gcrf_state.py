# /// script
# dependencies = ["brahe"]
# ///
"""
Transform state vector (position and velocity) from ITRF to GCRF
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

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

# Convert to GCRF Cartesian state
state_gcrf = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Define epoch
epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)
print(f"Epoch: {epc}")
print("GCRF state vector:")
print(f"  Position: [{state_gcrf[0]:.3f}, {state_gcrf[1]:.3f}, {state_gcrf[2]:.3f}] m")
print(
    f"  Velocity: [{state_gcrf[3]:.6f}, {state_gcrf[4]:.6f}, {state_gcrf[5]:.6f}] m/s\n"
)

# Transform to ITRF
state_itrf = bh.state_gcrf_to_itrf(epc, state_gcrf)

print("ITRF state vector:")
print(f"  Position: [{state_itrf[0]:.3f}, {state_itrf[1]:.3f}, {state_itrf[2]:.3f}] m")
print(
    f"  Velocity: [{state_itrf[3]:.6f}, {state_itrf[4]:.6f}, {state_itrf[5]:.6f}] m/s\n"
)

# Transform back to GCRF
state_gcrf_back = bh.state_itrf_to_gcrf(epc, state_itrf)

print("\nGCRF state vector (transformed from ITRF):")
print(
    f"  Position: [{state_gcrf_back[0]:.3f}, {state_gcrf_back[1]:.3f}, {state_gcrf_back[2]:.3f}] m"
)
print(
    f"  Velocity: [{state_gcrf_back[3]:.6f}, {state_gcrf_back[4]:.6f}, {state_gcrf_back[5]:.6f}] m/s"
)

diff_pos = np.linalg.norm(state_gcrf[0:3] - state_gcrf_back[0:3])
diff_vel = np.linalg.norm(state_gcrf[3:6] - state_gcrf_back[3:6])
print("\nRound-trip error:")
print(f"  Position: {diff_pos:.6e} m")
print(f"  Velocity: {diff_vel:.6e} m/s")
