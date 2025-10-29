# /// script
# dependencies = ["brahe"]
# ///
"""
Transform state vector (position and velocity) from ECI to ECEF
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

# Convert to ECI Cartesian state
state_eci = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)

print("ECI state vector:")
print(f"  Position: [{state_eci[0]:.3f}, {state_eci[1]:.3f}, {state_eci[2]:.3f}] m")
print(f"  Velocity: [{state_eci[3]:.6f}, {state_eci[4]:.6f}, {state_eci[5]:.6f}] m/s\n")

# ECI state vector:
#   Position: [1848964.106, -434937.468, 6560410.530] m
#   Velocity: [-7098.379734, -2173.344867, 1913.333385] m/s

# Transform to ECEF at specific epoch
state_ecef = bh.state_eci_to_ecef(epc, state_eci)

print("\nECEF state vector:")
print(f"  Position: [{state_ecef[0]:.3f}, {state_ecef[1]:.3f}, {state_ecef[2]:.3f}] m")
print(
    f"  Velocity: [{state_ecef[3]:.6f}, {state_ecef[4]:.6f}, {state_ecef[5]:.6f}] m/s"
)
# ECEF state vector:
#   Position: [757164.267, 1725863.563, 6564672.302] m
#   Velocity: [989.350643, -7432.740021, 1896.768934] m/s
