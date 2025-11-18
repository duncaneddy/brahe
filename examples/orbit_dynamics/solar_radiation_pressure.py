# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute solar radiation pressure acceleration with Earth shadow
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create an epoch (summer solstice for interesting Sun geometry)
epoch = bh.Epoch.from_datetime(2024, 6, 21, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Define satellite position (GEO satellite)
a = bh.R_EARTH + 35786e3  # Semi-major axis (m) - geostationary
e = 0.0001  # Near-circular
i = np.radians(0.1)  # Near-equatorial
raan = np.radians(0.0)  # RAAN (rad)
argp = np.radians(0.0)  # Argument of perigee (rad)
nu = np.radians(0.0)  # True anomaly (rad)

# Convert to Cartesian state
oe = np.array([a, e, i, raan, argp, nu])
state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
r_sat = state[0:3]  # Position vector (m)

print("Satellite position (ECI, m):")
print(f"  x = {r_sat[0] / 1e3:.1f} km")
print(f"  y = {r_sat[1] / 1e3:.1f} km")
print(f"  z = {r_sat[2] / 1e3:.1f} km")
print(f"  Altitude: {(np.linalg.norm(r_sat) - bh.R_EARTH) / 1e3:.1f} km")

# Get Sun position
r_sun = bh.sun_position(epoch)

print("\nSun position (ECI, AU):")
print(f"  x = {r_sun[0] / 1.496e11:.6f} AU")
print(f"  y = {r_sun[1] / 1.496e11:.6f} AU")
print(f"  z = {r_sun[2] / 1.496e11:.6f} AU")

# Eclipse condition - check shadow using both models
nu_conical = bh.eclipse_conical(r_sat, r_sun)
nu_cylindrical = bh.eclipse_cylindrical(r_sat, r_sun)

print("\nEclipse status:")
print(f"  Conical model: {nu_conical:.6f}")
print(f"  Cylindrical model: {nu_cylindrical:.6f}")

if nu_conical == 0.0:
    print("  Status: Full shadow (umbra)")
elif nu_conical == 1.0:
    print("  Status: Full sunlight")
else:
    print(f"  Status: Penumbra ({nu_conical * 100:.1f}% illuminated)")

# Define satellite SRP properties
mass = 1500.0  # kg (typical GEO satellite)
cr = 1.3  # Radiation pressure coefficient
area = 20.0  # m² (effective area - solar panels + body)
p0 = 4.56e-6  # Solar radiation pressure at 1 AU (N/m²)

print("\nSatellite SRP properties:")
print(f"  Mass: {mass:.1f} kg")
print(f"  Area: {area:.1f} m²")
print(f"  Cr coefficient: {cr:.1f}")
print(f"  Area/mass ratio: {area / mass:.6f} m²/kg")

# Compute solar radiation pressure acceleration
accel_srp = bh.accel_solar_radiation_pressure(r_sat, r_sun, mass, cr, area, p0)

print("\nSolar radiation pressure acceleration (ECI, m/s²):")
print(f"  ax = {accel_srp[0]:.12f}")
print(f"  ay = {accel_srp[1]:.12f}")
print(f"  az = {accel_srp[2]:.12f}")
print(f"  Magnitude: {np.linalg.norm(accel_srp):.12f} m/s²")

# Theoretical maximum (no eclipse)
accel_max = p0 * cr * area / mass
print(f"\nTheoretical maximum (full sun): {accel_max:.12f} m/s²")
print(f"Actual/Maximum ratio: {np.linalg.norm(accel_srp) / accel_max:.6f}")

# Compare to other forces at GEO
r_mag = np.linalg.norm(r_sat)
accel_gravity = bh.GM_EARTH / r_mag**2
print("\nFor comparison at GEO altitude:")
print(f"  Point-mass gravity: {accel_gravity:.9f} m/s²")
print(f"  SRP/Gravity ratio: {np.linalg.norm(accel_srp) / accel_gravity:.2e}")

# Expected output:
# Satellite position (ECI, m):
#   x = 42159.9 km
#   y = 0.0 km
#   z = 0.0 km
#   Altitude: 35781.8 km

# Sun position (ECI, AU):
#   x = -0.003352 AU
#   y = 0.932401 AU
#   z = 0.404245 AU

# Eclipse status:
#   Conical model: 1.000000
#   Cylindrical model: 1.000000
#   Status: Full sunlight

# Satellite SRP properties:
#   Mass: 1500.0 kg
#   Area: 20.0 m²
#   Cr coefficient: 1.3
#   Area/mass ratio: 0.013333 m²/kg

# Solar radiation pressure acceleration (ECI, m/s²):
#   ax = 0.000000000274
#   ay = -0.000000070212
#   az = -0.000000030441
#   Magnitude: 0.000000076528 m/s²

# Theoretical maximum (full sun): 0.000000079040 m/s²
# Actual/Maximum ratio: 0.968216

# For comparison at GEO altitude:
#   Point-mass gravity: 0.224252979 m/s²
#   SRP/Gravity ratio: 3.41e-07
