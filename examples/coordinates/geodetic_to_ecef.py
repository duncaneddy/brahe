# /// script
# dependencies = ["brahe"]
# ///
"""
Convert between geodetic (WGS84 ellipsoid) and ECEF Cartesian coordinates
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define a location in geodetic coordinates (WGS84 ellipsoid model)
# Boulder, Colorado (approximately)
lon = -122.4194  # Longitude (deg)
lat = 37.7749  # Latitude (deg)
alt = 16.0  # Altitude above WGS84 ellipsoid (m)

print("Geodetic coordinates (WGS84 ellipsoid model):")
print(f"Longitude: {lon:.4f}° = {np.radians(lon):.6f} rad")
print(f"Latitude:  {lat:.4f}° = {np.radians(lat):.6f} rad")
print(f"Altitude:  {alt:.1f} m\n")
# Longitude: -122.4194° = -2.136622 rad
# Latitude:  37.7749° = 0.659296 rad
# Altitude:  16.0 m

# Convert geodetic to ECEF Cartesian
geodetic = np.array([lon, lat, alt])
ecef = bh.position_geodetic_to_ecef(geodetic, bh.AngleFormat.DEGREES)

print("ECEF Cartesian coordinates:")
print(f"x = {ecef[0]:.3f} m")
print(f"y = {ecef[1]:.3f} m")
print(f"z = {ecef[2]:.3f} m")
print(f"Distance from Earth center: {np.linalg.norm(ecef):.3f} m\n")
# x = -2706181.627 m
# y = -4261070.165 m
# z = 3885735.291 m
# Distance from Earth center: 6370170.853 m
