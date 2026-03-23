# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates basic SpaceTrack query construction for GP and SATCAT data.
"""

import brahe as bh

# Build a GP query for the ISS by NORAD catalog ID
query = bh.SpaceTrackQuery(bh.RequestClass.GP).filter("NORAD_CAT_ID", "25544")

url_path = query.build()
print(f"GP query URL path:\n  {url_path}")

# Build a SATCAT query for US-owned objects
query = bh.SpaceTrackQuery(bh.RequestClass.SATCAT).filter("COUNTRY", "US")

url_path = query.build()
print(f"\nSATCAT query URL path:\n  {url_path}")

# The default controller is inferred from the request class
query = bh.SpaceTrackQuery(bh.RequestClass.CDM_PUBLIC)
url_path = query.build()
print(f"\nCDM query URL path (uses expandedspacedata controller):\n  {url_path}")
