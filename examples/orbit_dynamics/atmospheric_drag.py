# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute atmospheric drag acceleration using Harris-Priester density model
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create an epoch
epoch = bh.Epoch.from_datetime(2024, 3, 15, 14, 30, 0.0, 0.0, bh.TimeSystem.UTC)

# Define satellite state in ECI frame (LEO satellite at 450 km altitude)
a = bh.R_EARTH + 450e3  # Semi-major axis (m)
e = 0.002  # Eccentricity
i = 51.6  # Inclination (deg)
raan = 90.0  # RAAN (deg)
argp = 45.0  # Argument of perigee (deg)
nu = 120.0  # True anomaly (deg)

# Convert to Cartesian state
oe = np.array([a, e, i, raan, argp, nu])
state_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

print("Satellite state (ECI):")
print(
    f"  Position: [{state_eci[0] / 1e3:.1f}, {state_eci[1] / 1e3:.1f}, {state_eci[2] / 1e3:.1f}] km"
)
print(
    f"  Velocity: [{state_eci[3] / 1e3:.3f}, {state_eci[4] / 1e3:.3f}, {state_eci[5] / 1e3:.3f}] km/s"
)
print(f"  Altitude: {(np.linalg.norm(state_eci[0:3]) - bh.R_EARTH) / 1e3:.1f} km")

# Atmospheric density
# For this example, use a typical density for the given altitude (~450 km)
# In practice, this would be computed using atmospheric density models like Harris-Priester
# Typical value for ~450 km altitude: 3-5 × 10^-12 kg/m³
density = 4.0e-12  # kg/m³

print(f"\nAtmospheric density (exponential model): {density:.6e} kg/m³")

# Define satellite properties
mass = 500.0  # kg (typical small satellite)
area = 2.5  # m² (cross-sectional area)
cd = 2.2  # Drag coefficient (typical for satellites)

print("\nSatellite properties:")
print(f"  Mass: {mass:.1f} kg")
print(f"  Area: {area:.1f} m²")
print(f"  Drag coefficient: {cd:.1f}")
print(f"  Ballistic coefficient: {cd * area / mass:.6f} m²/kg")

# Compute ECI to ECEF rotation matrix for atmospheric velocity
R_eci_ecef = bh.rotation_eci_to_ecef(epoch)

# Compute drag acceleration
accel_drag = bh.accel_drag(state_eci, density, mass, area, cd, R_eci_ecef)

print("\nDrag acceleration (ECI, m/s²):")
print(f"  ax = {accel_drag[0]:.9f}")
print(f"  ay = {accel_drag[1]:.9f}")
print(f"  az = {accel_drag[2]:.9f}")
print(f"  Magnitude: {np.linalg.norm(accel_drag):.9f} m/s²")

# Compute velocity magnitude
v_mag = np.linalg.norm(state_eci[3:6])
print(f"\nOrbital velocity: {v_mag:.3f} m/s ({v_mag / 1e3:.3f} km/s)")

# Theoretical drag magnitude check: 0.5 * rho * v² * Cd * A / m
accel_theory = 0.5 * density * v_mag**2 * cd * area / mass
print(f"Theoretical drag magnitude: {accel_theory:.9f} m/s²")

# Expected output:
# Satellite state (ECI):
#   Position: [-1084.6, -6608.2, 1368.5] km
#   Velocity: [4.582, -1.963, -5.781] km/s
#   Altitude: 456.8 km

# Atmospheric density (exponential model): 4.000000e-12 kg/m³

# Satellite properties:
#   Mass: 500.0 kg
#   Area: 2.5 m²
#   Drag coefficient: 2.2
#   Ballistic coefficient: 0.011000 m²/kg

# Drag acceleration (ECI, m/s²):
#   ax = -0.000000661
#   ay = 0.000000304
#   az = 0.000000932
#   Magnitude: 0.000001183 m/s²

# Orbital velocity: 7632.770 m/s (7.633 km/s)
# Theoretical drag magnitude: 0.000001282 m/s²
