# /// script
# dependencies = ["brahe"]
# ///
"""
Convert ENZ (East-North-Zenith) relative position to ECEF coordinates
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define ground station location in geodetic coordinates
# Stanford University: (lon=-122.17329°, lat=37.42692°, alt=32.0m)
lon_deg = -122.17329
lat_deg = 37.42692
alt_m = 32.0

print("Ground Station (Stanford):")
print(f"Longitude: {lon_deg:.5f}° = {np.radians(lon_deg):.6f} rad")
print(f"Latitude:  {lat_deg:.5f}° = {np.radians(lat_deg):.6f} rad")
print(f"Altitude:  {alt_m:.1f} m\n")
# Longitude: -122.17329° = -2.132605 rad
# Latitude:  37.42692° = 0.653131 rad
# Altitude:  32.0 m

# Convert ground station to ECEF
geodetic_station = np.array([lon_deg, lat_deg, alt_m])
station_ecef = bh.position_geodetic_to_ecef(geodetic_station, bh.AngleFormat.DEGREES)

print("Ground Station ECEF:")
print(f"x = {station_ecef[0]:.3f} m")
print(f"y = {station_ecef[1]:.3f} m")
print(f"z = {station_ecef[2]:.3f} m\n")
# x = -2700691.122 m
# y = -4292566.016 m
# z = 3855395.780 m

# Define relative position in ENZ coordinates
# Example: 50 km East, 100 km North, 200 km Up from station
enz = np.array([50e3, 100e3, 200e3])

print("Relative position in ENZ frame:")
print(f"East:   {enz[0] / 1000:.1f} km")
print(f"North:  {enz[1] / 1000:.1f} km")
print(f"Zenith: {enz[2] / 1000:.1f} km\n")
# East:   50.0 km
# North:  100.0 km
# Zenith: 200.0 km

# Convert ENZ relative position to absolute ECEF position
target_ecef = bh.relative_position_enz_to_ecef(
    station_ecef, enz, bh.EllipsoidalConversionType.GEODETIC
)

print("Target position in ECEF:")
print(f"x = {target_ecef[0]:.3f} m")
print(f"y = {target_ecef[1]:.3f} m")
print(f"z = {target_ecef[2]:.3f} m")
print(f"Distance from Earth center: {np.linalg.norm(target_ecef) / 1000:.3f} km")
