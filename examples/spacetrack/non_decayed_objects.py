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

# Filter to only active payloads (exclude debris and rocket bodies)
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("DECAY_DATE", op.null_val())
    .filter("OBJECT_TYPE", "PAYLOAD")
    .order_by("NORAD_CAT_ID", bh.SortOrder.ASC)
)

url_path = query.build()
print(f"\nActive payloads only:\n  {url_path}")

# Filter to active objects in LEO (period under 128 minutes)
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("DECAY_DATE", op.null_val())
    .filter("PERIOD", op.less_than("128"))
    .order_by("NORAD_CAT_ID", bh.SortOrder.ASC)
)

url_path = query.build()
print(f"\nActive LEO objects:\n  {url_path}")
