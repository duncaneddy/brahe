# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates querying the most recent GP ephemeris for a single object.
"""

import brahe as bh

# Get the latest GP record for the ISS (NORAD 25544)
# Order by EPOCH descending so the most recent is first, limit to 1
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("NORAD_CAT_ID", "25544")
    .order_by("EPOCH", bh.SortOrder.DESC)
    .limit(1)
)

url_path = query.build()
print(f"Latest GP for ISS:\n  {url_path}")
# Latest GP for ISS:
#   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544/orderby/EPOCH desc/limit/1/format/json

# Get the latest GP for a Starlink satellite (NORAD 48274)
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("NORAD_CAT_ID", "48274")
    .order_by("EPOCH", bh.SortOrder.DESC)
    .limit(1)
)

url_path = query.build()
print(f"\nLatest GP for Starlink-2541:\n  {url_path}")
# Latest GP for Starlink-2541:
#   /basicspacedata/query/class/gp/NORAD_CAT_ID/48274/orderby/EPOCH desc/limit/1/format/json
