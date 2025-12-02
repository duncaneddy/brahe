# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates pattern matching in SpaceTrack queries.
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

    # Like operator: pattern matching with % wildcard
    # Find all Starlink satellites
    starlinks = client.gp(
        object_name="~~STARLINK%",
        limit=10,
    )
    print(f"Found {len(starlinks)} Starlink satellites")

    # Starts with: international designator prefix
    # Find objects from 2024 launches
    launches_2024 = client.satcat(
        intldes="^2024-",
        limit=10,
    )
    print(f"Found {len(launches_2024)} objects from 2024 launches")
    # Found 10 Starlink satellites
    # Found 10 objects from 2024 launches

print("Example completed successfully!")
