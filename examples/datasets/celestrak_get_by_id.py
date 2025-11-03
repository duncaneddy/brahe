# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Get a single satellite TLE by NORAD ID from CelesTrak.

This example demonstrates the cache-efficient pattern: providing the group name
allows brahe to use cached group data rather than making a new API request.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Get ISS TLE by NORAD ID
# The group hint ("stations") allows brahe to check cached data first
name, line1, line2 = bh.datasets.celestrak.get_tle_by_id(25544, "stations")

# Parse TLE data to get epoch and orbital elements
epoch, oe = bh.keplerian_elements_from_tle(line1, line2)

print("ISS TLE:")
print(f"  Name: {name}")
print(f"  Epoch: {epoch}")
print(f"  Inclination: {oe[2]:.2f}°")
print(f"  RAAN: {oe[3]:.2f}°")
print(f"  Eccentricity: {oe[1]:.6f}")

# Expected output:
# ISS TLE:
#   Name: ISS (ZARYA)
#   Epoch: 2025-11-02 10:09:34.283 UTC
#   Inclination: 51.63°
#   RAAN: 342.07°
#   Eccentricity: 0.000497
