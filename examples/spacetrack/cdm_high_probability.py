# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates querying Conjunction Data Messages (CDMs) from Space-Track.
"""

import brahe as bh
from brahe.spacetrack import operators as op

# Query high-probability conjunction events (Pc > 1e-3)
# CDMPublic uses the expandedspacedata controller automatically
query = (
    bh.SpaceTrackQuery(bh.RequestClass.CDM_PUBLIC)
    .filter("PC", op.greater_than("1.0e-3"))
    .order_by("TCA", bh.SortOrder.DESC)
    .limit(25)
)

url_path = query.build()
print(f"High-probability CDMs:\n  {url_path}")
# High-probability CDMs:
#   /expandedspacedata/query/class/cdm_public/PC/>1.0e-3/orderby/TCA desc/limit/25/format/json

# Query CDMs for a specific satellite (e.g., ISS, NORAD 25544)
query = (
    bh.SpaceTrackQuery(bh.RequestClass.CDM_PUBLIC)
    .filter("SAT_1_ID", "25544")
    .order_by("TCA", bh.SortOrder.DESC)
    .limit(10)
)

url_path = query.build()
print(f"\nCDMs involving ISS:\n  {url_path}")
# CDMs involving ISS:
#   /expandedspacedata/query/class/cdm_public/SAT_1_ID/25544/orderby/TCA desc/limit/10/format/json

# Query upcoming conjunctions within the next 7 days
query = (
    bh.SpaceTrackQuery(bh.RequestClass.CDM_PUBLIC)
    .filter("TCA", op.inclusive_range(op.now(), op.now_offset(7)))
    .order_by("TCA", bh.SortOrder.ASC)
)

url_path = query.build()
print(f"\nUpcoming conjunctions (next 7 days):\n  {url_path}")
# Upcoming conjunctions (next 7 days):
#   /expandedspacedata/query/class/cdm_public/TCA/now--now+7/orderby/TCA asc/format/json
