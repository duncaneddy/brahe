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
records = client.get_gp(catnr=25544)
record = records[0]

print("ISS GP Data:")
print(f"  Name: {record.object_name}")
print(f"  NORAD ID: {record.norad_cat_id}")
print(f"  Epoch: {record.epoch}")
print(f"  Inclination: {record.inclination:.2f}°")
print(f"  RAAN: {record.ra_of_asc_node:.2f}°")
print(f"  Eccentricity: {record.eccentricity:.6f}")
