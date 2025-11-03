# /// script
# dependencies = ["brahe"]
# ///
"""
Create a PointLocation from geodetic coordinates (longitude, latitude, altitude).
Demonstrates basic construction and naming.
"""

import brahe as bh

bh.initialize_eop()

# Create location (longitude, latitude, altitude in meters)
# San Francisco, CA
sf = bh.PointLocation(-122.4194, 37.7749, 0.0)

# Add an identifier for clarity
sf = sf.with_name("San Francisco")

print(f"Location: {sf.get_name()}")
print(f"Longitude: {sf.longitude(bh.AngleFormat.DEGREES):.4f} deg")
print(f"Latitude: {sf.latitude(bh.AngleFormat.DEGREES):.4f} deg")

# Expected output:
# Location: San Francisco
# Longitude: -122.4194 deg
# Latitude: 37.7749 deg
