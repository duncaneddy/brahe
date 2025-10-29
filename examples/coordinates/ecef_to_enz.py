# /// script
# dependencies = ["brahe"]
# ///
"""
Convert satellite position from ECEF to ENZ (East-North-Zenith) coordinates relative to a ground station
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

# Define satellite in sun-synchronous orbit at 500 km altitude
# SSO orbit passes over Stanford at approximately 10:30 AM local time
# Orbital elements: [a, e, i, RAAN, omega, M]
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 240.0, 0.0, 90.0])

# Define epoch when satellite passes near Stanford (Jan 1, 2024, 17:05 UTC)
epoch = bh.Epoch.from_datetime(2024, 1, 1, 17, 5, 0.0, 0.0, bh.TimeSystem.UTC)

# Convert orbital elements to ECI state
sat_state_eci = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)

# Convert ECI state to ECEF at the given epoch
sat_state_ecef = bh.state_eci_to_ecef(epoch, sat_state_eci)
sat_ecef = sat_state_ecef[0:3]  # Extract position only

year, month, day, hour, minute, second, ns = epoch.to_datetime()
print(f"Epoch: {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f} UTC")
print("Satellite ECEF:")
print(f"x = {sat_ecef[0]:.3f} m")
print(f"y = {sat_ecef[1]:.3f} m")
print(f"z = {sat_ecef[2]:.3f} m\n")

# Convert satellite position to ENZ coordinates relative to ground station
enz = bh.relative_position_ecef_to_enz(
    station_ecef, sat_ecef, bh.EllipsoidalConversionType.GEODETIC
)

print("Satellite position in ENZ frame (relative to Stanford):")
print(f"East:   {enz[0] / 1000:.3f} km")
print(f"North:  {enz[1] / 1000:.3f} km")
print(f"Zenith: {enz[2] / 1000:.3f} km")
print(f"Range:  {np.linalg.norm(enz) / 1000:.3f} km")
