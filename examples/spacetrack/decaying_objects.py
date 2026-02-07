# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates querying decay predictions for objects expected to reenter soon.
"""

import brahe as bh
from brahe.spacetrack import operators as op

# Get objects predicted to decay within the next 30 days
# The Decay request class provides reentry predictions
query = (
    bh.SpaceTrackQuery(bh.RequestClass.DECAY)
    .filter("DECAY_EPOCH", op.inclusive_range(op.now(), op.now_offset(30)))
    .order_by("DECAY_EPOCH", bh.SortOrder.ASC)
)

url_path = query.build()
print(f"Decaying within 30 days:\n  {url_path}")
# Decaying within 30 days:
#   /basicspacedata/query/class/decay/DECAY_EPOCH/now--now+30/orderby/DECAY_EPOCH asc/format/json

# Get recent actual decays from the past 7 days
query = (
    bh.SpaceTrackQuery(bh.RequestClass.DECAY)
    .filter("DECAY_EPOCH", op.inclusive_range(op.now_offset(-7), op.now()))
    .filter("MSG_TYPE", "Decay")
    .order_by("DECAY_EPOCH", bh.SortOrder.DESC)
)

url_path = query.build()
print(f"\nRecent decays (last 7 days):\n  {url_path}")
# Recent decays (last 7 days):
#   /basicspacedata/query/class/decay/DECAY_EPOCH/now-7--now/MSG_TYPE/Decay/orderby/DECAY_EPOCH desc/format/json
