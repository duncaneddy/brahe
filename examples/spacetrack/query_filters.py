# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates using SpaceTrack operator functions to build filtered queries.
"""

import brahe as bh
from brahe.spacetrack import operators as op

# Filter by NORAD ID range using inclusive_range
query = bh.SpaceTrackQuery(bh.RequestClass.GP).filter(
    "NORAD_CAT_ID", op.inclusive_range("25544", "25600")
)
print(f"Range filter:\n  {query.build()}")
# Range filter:
#   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544--25600/format/json

# Filter for objects with low eccentricity using less_than
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("ECCENTRICITY", op.less_than("0.01"))
    .filter("OBJECT_TYPE", "PAYLOAD")
)
print(f"\nMultiple filters:\n  {query.build()}")
# Multiple filters:
#   /basicspacedata/query/class/gp/ECCENTRICITY/<0.01/OBJECT_TYPE/PAYLOAD/format/json

# Filter for recently launched objects using greater_than + now_offset
query = bh.SpaceTrackQuery(bh.RequestClass.SATCAT).filter(
    "LAUNCH", op.greater_than(op.now_offset(-30))
)
print(f"\nRecent launches (last 30 days):\n  {query.build()}")
# Recent launches (last 30 days):
#   /basicspacedata/query/class/satcat/LAUNCH/>now-30/format/json

# Search by name pattern using like
query = bh.SpaceTrackQuery(bh.RequestClass.SATCAT).filter(
    "SATNAME", op.like("STARLINK")
)
print(f"\nName pattern match:\n  {query.build()}")
# Name pattern match:
#   /basicspacedata/query/class/satcat/SATNAME/~~STARLINK/format/json

# Filter for multiple NORAD IDs using or_list
query = bh.SpaceTrackQuery(bh.RequestClass.GP).filter(
    "NORAD_CAT_ID", op.or_list(["25544", "48274", "54216"])
)
print(f"\nMultiple IDs:\n  {query.build()}")
# Multiple IDs:
#   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544,48274,54216/format/json
