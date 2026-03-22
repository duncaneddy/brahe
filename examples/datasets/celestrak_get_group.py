# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Get GP data for a satellite group from CelesTrak.

This example demonstrates the most efficient way to download GP data:
getting entire groups rather than individual satellites.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Download GP data for the Starlink group
# This fetches all Starlink satellites in one request
client = bh.celestrak.CelestrakClient()
records = client.get_gp(group="starlink")

print(f"Downloaded {len(records)} Starlink GP records")

# Each record has orbital elements and metadata
record = records[0]
print("\nFirst record:")
print(f"  Name: {record.object_name}")
print(f"  NORAD ID: {record.norad_cat_id}")
print(f"  Epoch: {record.epoch}")
print(f"  Inclination: {record.inclination:.2f}°")
print(f"  Eccentricity: {record.eccentricity:.6f}")
