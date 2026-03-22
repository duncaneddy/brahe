# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Get satellite GP data by name from CelesTrak.

This example demonstrates searching for satellites by name using the
CelestrakQuery builder's name_search method.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Search by name
client = bh.celestrak.CelestrakClient()
records = client.get_gp(name="ISS")

print(f"Found {len(records)} results for 'ISS'")
for record in records[:5]:
    print(f"  {record.object_name} (NORAD ID: {record.norad_cat_id})")

# The first result should be ISS (ZARYA)
iss = records[0]
print("\nISS GP Data:")
print(f"  Name: {iss.object_name}")
print(f"  NORAD ID: {iss.norad_cat_id}")
print(f"  Epoch: {iss.epoch}")
print(f"  Inclination: {iss.inclination:.2f}°")
print(f"  Eccentricity: {iss.eccentricity:.6f}")
