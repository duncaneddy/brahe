# /// script
# dependencies = ["brahe"]
# ///
"""
Transform state vector (position and velocity) from ECEF to ECI
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

# Convert to ECI Cartesian state
state_eci = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)

# Define epoch
epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)
print(f"Epoch: {epc}")
print("ECI state vector:")
print(f"  Position: [{state_eci[0]:.3f}, {state_eci[1]:.3f}, {state_eci[2]:.3f}] m")
print(f"  Velocity: [{state_eci[3]:.6f}, {state_eci[4]:.6f}, {state_eci[5]:.6f}] m/s\n")

# Transform to ECEF
state_ecef = bh.state_eci_to_ecef(epc, state_eci)

print("ECEF state vector:")
print(f"  Position: [{state_ecef[0]:.3f}, {state_ecef[1]:.3f}, {state_ecef[2]:.3f}] m")
print(
    f"  Velocity: [{state_ecef[3]:.6f}, {state_ecef[4]:.6f}, {state_ecef[5]:.6f}] m/s\n"
)

# Transform back to ECI
state_eci_back = bh.state_ecef_to_eci(epc, state_ecef)

print("\nECI state vector (transformed from ECEF):")
print(
    f"  Position: [{state_eci_back[0]:.3f}, {state_eci_back[1]:.3f}, {state_eci_back[2]:.3f}] m"
)
print(
    f"  Velocity: [{state_eci_back[3]:.6f}, {state_eci_back[4]:.6f}, {state_eci_back[5]:.6f}] m/s"
)

# Verify round-trip transformation
diff_pos = np.linalg.norm(state_eci[0:3] - state_eci_back[0:3])
diff_vel = np.linalg.norm(state_eci[3:6] - state_eci_back[3:6])
print("\nRound-trip error:")
print(f"  Position: {diff_pos:.6e} m")
print(f"  Velocity: {diff_vel:.6e} m/s")

# Expected output:
# Position: [1848964.106, -434937.468, 6560410.530] m
# Velocity: [-7098.379734, -2173.344867, 1913.333385] m/s
# Round-trip error: ~1e-9 m, ~1e-12 m/s
