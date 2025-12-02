# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates ordering results in SpaceTrack queries.
"""

import os

import brahe as bh

# Get credentials from environment
# Note: For production use, most users should use the default constructor:
#   client = bh.SpaceTrackClient(user, password)
# These examples use the test server for automated testing.
user = os.environ.get("TEST_SPACETRACK_USER", "")
password = os.environ.get("TEST_SPACETRACK_PASS", "")

if not user or not password:
    print("Set TEST_SPACETRACK_USER and TEST_SPACETRACK_PASS environment variables")
else:
    client = bh.SpaceTrackClient(
        user, password, base_url="https://for-testing-only.space-track.org/"
    )

    # Order by epoch descending (most recent first)
    recent_iss = client.gp(
        norad_cat_id=25544,
        orderby="EPOCH desc",
        limit=5,
    )
    print("ISS GP records (most recent first):")
    for record in recent_iss:
        print(f"  Epoch: {record.epoch}")

    # Order by launch date ascending (oldest first)
    oldest_sats = client.satcat(
        current="Y",
        orderby="LAUNCH asc",
        limit=5,
    )
    print("\nOldest satellites still in orbit:")
    for sat in oldest_sats:
        print(f"  {sat.object_name}: launched {sat.launch}")
    # ISS GP records (most recent first):
    #   Epoch: 2024-01-15 12:00:00
    #   Epoch: 2024-01-14 12:00:00
    #   ...
    # Oldest satellites still in orbit:
    #   VANGUARD 1: launched 1958-03-17
    #   ...

print("Example completed successfully!")
