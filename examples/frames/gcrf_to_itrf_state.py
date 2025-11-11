# /// script
# dependencies = ["brahe"]
# ///
"""
Transform state vector (position and velocity) from GCRF to ITRF
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
# Orbital elements (degrees):
#   a    = 6878136.300 m = 500.0 km altitude
#   e    = 0.0100
#   i    = 97.8000°
#   Ω    = 15.0000°
#   ω    = 30.0000°
#   M    = 45.0000°


epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)
print(f"Epoch: {epc}")
# Epoch: 2024-01-01 12:00:00.000 UTC

# Convert to GCRF Cartesian state
state_gcrf = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)

print("GCRF state vector:")
print(f"  Position: [{state_gcrf[0]:.3f}, {state_gcrf[1]:.3f}, {state_gcrf[2]:.3f}] m")
print(
    f"  Velocity: [{state_gcrf[3]:.6f}, {state_gcrf[4]:.6f}, {state_gcrf[5]:.6f}] m/s\n"
)

# GCRF state vector:
#   Position: [1848964.106, -434937.468, 6560410.530] m
#   Velocity: [-7098.379734, -2173.344867, 1913.333385] m/s

# Transform to ITRF at specific epoch
state_itrf = bh.state_gcrf_to_itrf(epc, state_gcrf)

print("\nITRF state vector:")
print(f"  Position: [{state_itrf[0]:.3f}, {state_itrf[1]:.3f}, {state_itrf[2]:.3f}] m")
print(
    f"  Velocity: [{state_itrf[3]:.6f}, {state_itrf[4]:.6f}, {state_itrf[5]:.6f}] m/s"
)
# ITRF state vector:
#   Position: [757164.267, 1725863.563, 6564672.302] m
#   Velocity: [989.350643, -7432.740021, 1896.768934] m/s
