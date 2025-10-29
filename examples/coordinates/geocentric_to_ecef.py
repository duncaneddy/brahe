# /// script
# dependencies = ["brahe"]
# ///
"""
Convert between geocentric spherical and ECEF Cartesian coordinates
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define a location in geocentric coordinates (spherical Earth model)
# Boulder, Colorado (approximately)
lon = -122.4194  # Longitude (deg)
lat = 37.7749  # Latitude (deg)
alt = 13.8  # Altitude above spherical Earth surface (m)

print("Geocentric coordinates (spherical Earth model):")
print(f"Longitude: {lon:.4f}° = {np.radians(lon):.6f} rad")
print(f"Latitude:  {lat:.4f}° = {np.radians(lat):.6f} rad")
print(f"Altitude:  {alt:.1f} m\n")
# Longitude: -122.4194° = -2.136622 rad
# Latitude:  37.7749° = 0.659296 rad
# Altitude:  13.8 m

# Convert geocentric to ECEF Cartesian
geocentric = np.array([lon, lat, alt])
ecef = bh.position_geocentric_to_ecef(geocentric, bh.AngleFormat.DEGREES)

print("ECEF Cartesian coordinates:")
print(f"x = {ecef[0]:.3f} m")
print(f"y = {ecef[1]:.3f} m")
print(f"z = {ecef[2]:.3f} m")
print(f"Distance from Earth center: {np.linalg.norm(ecef):.3f} m\n")
# x = -2702779.686 m
# y = -4255713.575 m
# z = 3907005.447 m
# Distance from Earth center: 6378150.800 m
