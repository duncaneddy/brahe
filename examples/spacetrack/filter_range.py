# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates range queries in SpaceTrack.
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

    # Date range: objects launched in January 2024
    jan_launches = client.satcat(
        launch="2024-01-01--2024-01-31",
        limit=10,
    )
    print(f"Found {len(jan_launches)} objects launched in January 2024")

    # Epoch range for GP data
    gp_range = client.gp(
        epoch="2024-01-01--2024-01-07",
        norad_cat_id=25544,
        limit=10,
    )
    print(f"Found {len(gp_range)} ISS GP records from first week of 2024")
    # Found 10 objects launched in January 2024
    # Found 10 ISS GP records from first week of 2024

print("Example completed successfully!")
