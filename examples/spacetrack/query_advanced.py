# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates advanced SpaceTrack query options: ordering, limits, and predicates.
"""

import brahe as bh

# Order results by epoch descending and limit to 5 records
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("NORAD_CAT_ID", "25544")
    .order_by("EPOCH", bh.SortOrder.DESC)
    .limit(5)
)
print(f"Ordered and limited:\n  {query.build()}")
# Ordered and limited:
#   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544/orderby/EPOCH desc/limit/5/format/json

# Use limit with offset for pagination
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("OBJECT_TYPE", "PAYLOAD")
    .order_by("NORAD_CAT_ID", bh.SortOrder.ASC)
    .limit_offset(10, 20)
)
print(f"\nPaginated results:\n  {query.build()}")
# Paginated results:
#   /basicspacedata/query/class/gp/OBJECT_TYPE/PAYLOAD/orderby/NORAD_CAT_ID asc/limit/10,20/format/json

# Select specific fields with predicates_filter
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("NORAD_CAT_ID", "25544")
    .predicates_filter(["OBJECT_NAME", "EPOCH", "INCLINATION", "PERIOD"])
)
print(f"\nFiltered fields:\n  {query.build()}")
# Filtered fields:
#   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544/predicates/OBJECT_NAME,EPOCH,INCLINATION,PERIOD/format/json

# Enable metadata and distinct results
query = (
    bh.SpaceTrackQuery(bh.RequestClass.SATCAT)
    .filter("COUNTRY", "US")
    .distinct(True)
    .metadata(True)
)
print(f"\nDistinct with metadata:\n  {query.build()}")
# Distinct with metadata:
#   /basicspacedata/query/class/satcat/COUNTRY/US/metadata/true/distinct/true/format/json
