# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Search and filter the GCAT SATCAT catalog.

This example demonstrates name search and filter chaining to narrow
down the catalog to specific subsets of objects.
"""

import brahe as bh

# Download the SATCAT catalog
satcat = bh.datasets.gcat.get_satcat()
print(f"Total records: {len(satcat)}")

# Search by name (case-insensitive, searches both name and pl_name)
starlink = satcat.search_by_name("starlink")
print(f"\nStarlink name search: {len(starlink)} results")

# Filter chaining: payloads that are operational in LEO
payloads = satcat.filter_by_type("P")
print(f"\nAll payloads: {len(payloads)}")

operational = payloads.filter_by_status("O")
print(f"Operational payloads: {len(operational)}")

leo = operational.filter_by_perigee_range(160.0, 2000.0)
print(f"Operational LEO payloads: {len(leo)}")

# Filter by inclination range (sun-synchronous orbits ~96-99 deg)
sso = operational.filter_by_inc_range(96.0, 99.0)
print(f"Operational SSO payloads: {len(sso)}")
