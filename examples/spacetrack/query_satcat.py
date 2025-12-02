# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates querying satellite catalog (SATCAT) data from SpaceTrack.
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

    # Query SATCAT for ISS
    records = client.satcat(norad_cat_id=25544, limit=1)

    # Records are returned as SATCATRecord objects with attribute access
    record = records[0]
    print(f"Name: {record.object_name}")
    print(f"NORAD ID: {record.norad_cat_id}")
    print(f"Intl Designator: {record.intldes}")
    print(f"Launch Date: {record.launch}")
    print(f"Country: {record.country}")
    print(f"Object Type: {record.object_type}")
    # Name: ISS (ZARYA)
    # NORAD ID: 25544
    # Intl Designator: 1998-067A
    # Launch Date: 1998-11-20
    # Country: ISS
    # Object Type: PAYLOAD

print("Example completed successfully!")
