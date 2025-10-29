# /// script
# dependencies = ["brahe"]
# ///
"""
Compute sun-synchronous inclination for an orbit.

This example demonstrates computing the required inclination for a
sun-synchronous orbit at various altitudes and eccentricities. Sun-synchronous
orbits maintain a constant angle relative to the Sun, useful for Earth
observation missions requiring consistent lighting conditions.
"""

import brahe as bh

bh.initialize_eop()

# Example 1: Typical sun-synchronous LEO at 800 km altitude
a_leo = bh.R_EARTH + 800e3  # Semi-major axis
e_leo = 0.0  # Circular orbit

inc_leo_deg = bh.sun_synchronous_inclination(a_leo, e_leo, bh.AngleFormat.DEGREES)
inc_leo_rad = bh.sun_synchronous_inclination(a_leo, e_leo, bh.AngleFormat.RADIANS)

print("Sun-synchronous LEO (800 km, circular):")
print(f"  Inclination: {inc_leo_deg:.3f} degrees")
print(f"  Inclination: {inc_leo_rad:.6f} radians")

# Example 2: Different altitudes
altitudes = [500, 600, 700, 800, 900, 1000]  # km
print("\nSun-synchronous inclination vs altitude (circular orbits):")
for alt_km in altitudes:
    a = bh.R_EARTH + alt_km * 1e3
    inc = bh.sun_synchronous_inclination(a, 0.0, bh.AngleFormat.DEGREES)
    print(f"  {alt_km:4d} km: {inc:.3f} deg")

# Example 3: Effect of eccentricity
a_fixed = bh.R_EARTH + 700e3
eccentricities = [0.0, 0.001, 0.005, 0.01, 0.02]

print("\nSun-synchronous inclination vs eccentricity (700 km orbit):")
for e in eccentricities:
    inc = bh.sun_synchronous_inclination(a_fixed, e, bh.AngleFormat.DEGREES)
    print(f"  e = {e:.3f}: {inc:.3f} deg")

# Example 4: Practical mission example (Landsat-like)
a_landsat = bh.R_EARTH + 705e3
e_landsat = 0.0001
inc_landsat = bh.sun_synchronous_inclination(
    a_landsat, e_landsat, bh.AngleFormat.DEGREES
)

print("\nLandsat-like orbit (705 km, nearly circular):")
print(f"  Inclination: {inc_landsat:.3f} deg")
print(f"  Period: {bh.orbital_period(a_landsat) / 60:.3f} min")

# Expected output:
# Sun-synchronous LEO (800 km, circular):
#   Inclination: 98.603 degrees
#   Inclination: 1.720948 radians

# Sun-synchronous inclination vs altitude (circular orbits):
#    500 km: 97.402 deg
#    600 km: 97.788 deg
#    700 km: 98.188 deg
#    800 km: 98.603 deg
#    900 km: 99.033 deg
#   1000 km: 99.479 deg

# Sun-synchronous inclination vs eccentricity (700 km orbit):
#   e = 0.000: 98.188 deg
#   e = 0.001: 98.188 deg
#   e = 0.005: 98.187 deg
#   e = 0.010: 98.186 deg
#   e = 0.020: 98.181 deg

# Landsat-like orbit (705 km, nearly circular):
#   Inclination: 98.208 deg
#   Period: 98.878 min
