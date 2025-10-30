# /// script
# dependencies = ["brahe"]
# ///

"""
Parse a Two-Line Element (TLE) set to extract orbital elements.

This example demonstrates how to extract the epoch and Keplerian orbital
elements from a TLE set.
"""

import brahe as bh

# ISS TLE (NORAD ID 25544)
line1 = "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995"
line2 = "2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49579513535999"

# Parse TLE to extract epoch and orbital elements
epoch, elements = bh.keplerian_elements_from_tle(line1, line2)

# Extract individual orbital elements
# Note: Angles are returned in degrees (exception to library convention)
a = elements[0]  # Semi-major axis (m)
e = elements[1]  # Eccentricity
i = elements[2]  # Inclination (deg)
raan = elements[3]  # Right Ascension of Ascending Node (deg)
argp = elements[4]  # Argument of Periapsis (deg)
M = elements[5]  # Mean Anomaly (deg)

print(f"ISS Orbital Elements (Epoch: {epoch})")
print(f"  Semi-major axis: {a / 1000:.3f} km")
print(f"  Eccentricity: {e:.6f}")
print(f"  Inclination: {i:.4f} deg")
print(f"  RAAN: {raan:.4f} deg")
print(f"  Arg of Perigee: {argp:.4f} deg")
print(f"  Mean Anomaly: {M:.4f} deg")

# Expected output:
# ISS Orbital Elements (Epoch: 2025-10-29T11:44:55.766182400 UTC)
#   Semi-major axis: 6795.445 km
#   Eccentricity: 0.000481
#   Inclination: 51.6347 deg
#   RAAN: 1.5519 deg
#   Arg of Perigee: 353.3325 deg
#   Mean Anomaly: 6.7599 deg
