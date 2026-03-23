# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates setting different output formats on SpaceTrack queries.
"""

import brahe as bh

# Default format is JSON
query = bh.SpaceTrackQuery(bh.RequestClass.GP).filter("NORAD_CAT_ID", "25544")
print(f"Default (JSON):\n  {query.build()}")

# Request TLE format for direct TLE text output
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("NORAD_CAT_ID", "25544")
    .format(bh.OutputFormat.TLE)
)
print(f"\nTLE format:\n  {query.build()}")

# Request CSV format for spreadsheet-compatible output
query = (
    bh.SpaceTrackQuery(bh.RequestClass.SATCAT)
    .filter("COUNTRY", "US")
    .limit(10)
    .format(bh.OutputFormat.CSV)
)
print(f"\nCSV format:\n  {query.build()}")

# Request KVN (CCSDS Keyword-Value Notation) format
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("NORAD_CAT_ID", "25544")
    .format(bh.OutputFormat.KVN)
)
print(f"\nKVN format:\n  {query.build()}")
