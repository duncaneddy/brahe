# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates querying the latest GP data for all non-decayed (active) objects.
"""

import brahe as bh
from brahe.spacetrack import operators as op

# Get latest GP for all non-decayed objects
# DECAY_DATE = null-val means the object has not decayed
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("DECAY_DATE", op.null_val())
    .order_by("NORAD_CAT_ID", bh.SortOrder.ASC)
)

url_path = query.build()
print(f"All non-decayed objects:\n  {url_path}")
# All non-decayed objects:
#   /basicspacedata/query/class/gp/DECAY_DATE/null-val/orderby/NORAD_CAT_ID asc/format/json

# Filter to only active payloads (exclude debris and rocket bodies)
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("DECAY_DATE", op.null_val())
    .filter("OBJECT_TYPE", "PAYLOAD")
    .order_by("NORAD_CAT_ID", bh.SortOrder.ASC)
)

url_path = query.build()
print(f"\nActive payloads only:\n  {url_path}")
# Active payloads only:
#   /basicspacedata/query/class/gp/DECAY_DATE/null-val/OBJECT_TYPE/PAYLOAD/orderby/NORAD_CAT_ID asc/format/json

# Filter to active objects in LEO (period under 128 minutes)
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("DECAY_DATE", op.null_val())
    .filter("PERIOD", op.less_than("128"))
    .order_by("NORAD_CAT_ID", bh.SortOrder.ASC)
)

url_path = query.build()
print(f"\nActive LEO objects:\n  {url_path}")
# Active LEO objects:
#   /basicspacedata/query/class/gp/DECAY_DATE/null-val/PERIOD/<128/orderby/NORAD_CAT_ID asc/format/json
