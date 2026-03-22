# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrate various methods for accessing PointLocation coordinates.
Shows degree/radian accessors, geodetic arrays, and ECEF conversion.
"""

import brahe as bh

bh.initialize_eop()

location = bh.PointLocation(-122.4194, 37.7749, 0.0)

# Access in degrees
print(f"Longitude: {location.longitude(bh.AngleFormat.DEGREES)} deg")
print(f"Latitude: {location.latitude(bh.AngleFormat.DEGREES)} deg")
print(f"Altitude: {location.altitude()} m")

# Shorthand access (in degrees)
print(f"Lon (deg): {location.lon:.6f}")
print(f"Lat (deg): {location.lat:.6f}")

# Get geodetic array [lat, lon, alt] in radians and meters
geodetic = location.center_geodetic()
print(f"Geodetic: [{geodetic[0]:.6f}, {geodetic[1]:.6f}, {geodetic[2]:.1f}]")

# Get ECEF Cartesian position [x, y, z] in meters
ecef = location.center_ecef()
print(f"ECEF: [{ecef[0]:.1f}, {ecef[1]:.1f}, {ecef[2]:.1f}] m")
