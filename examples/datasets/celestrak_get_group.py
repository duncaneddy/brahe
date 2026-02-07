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
query = bh.celestrak.CelestrakQuery.gp().group("starlink")
records = client.query_gp(query)

print(f"Downloaded {len(records)} Starlink GP records")

# Each record has orbital elements and metadata
record = records[0]
print("\nFirst record:")
print(f"  Name: {record.object_name}")
print(f"  NORAD ID: {record.norad_cat_id}")
print(f"  Epoch: {record.epoch}")
print(f"  Inclination: {record.inclination:.2f}°")
print(f"  Eccentricity: {record.eccentricity:.6f}")

# Expected output:
# Downloaded 8647 Starlink GP records
#
# First record:
#   Name: STARLINK-1008
#   NORAD ID: 44714
#   Epoch: 2025-11-02T10:49:16.197504
#   Inclination: 53.05°
#   Eccentricity: 0.000137
