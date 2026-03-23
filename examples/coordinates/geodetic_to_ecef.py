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

# Convert geodetic to ECEF Cartesian
geodetic = np.array([lon, lat, alt])
ecef = bh.position_geodetic_to_ecef(geodetic, bh.AngleFormat.DEGREES)

print("ECEF Cartesian coordinates:")
print(f"x = {ecef[0]:.3f} m")
print(f"y = {ecef[1]:.3f} m")
print(f"z = {ecef[2]:.3f} m")
print(f"Distance from Earth center: {np.linalg.norm(ecef):.3f} m\n")
