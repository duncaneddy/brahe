# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Get a single satellite GP record by NORAD ID from CelesTrak.

This example demonstrates querying CelesTrak for a satellite's general
perturbations (GP) data using its NORAD catalog number.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Query ISS GP data by NORAD catalog number
client = bh.celestrak.CelestrakClient()
query = bh.celestrak.CelestrakQuery.gp().catnr(25544)
records = client.query_gp(query)
record = records[0]

print("ISS GP Data:")
print(f"  Name: {record.object_name}")
print(f"  NORAD ID: {record.norad_cat_id}")
print(f"  Epoch: {record.epoch}")
print(f"  Inclination: {record.inclination:.2f}°")
print(f"  RAAN: {record.ra_of_asc_node:.2f}°")
print(f"  Eccentricity: {record.eccentricity:.6f}")

# Expected output:
# ISS GP Data:
#   Name: ISS (ZARYA)
#   NORAD ID: 25544
#   Epoch: 2025-11-02T10:09:34.283392
#   Inclination: 51.63°
#   RAAN: 342.07°
#   Eccentricity: 0.000497
