# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates comparison operators in SpaceTrack queries.
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

    # Greater than: epochs after a date
    recent = client.gp(
        epoch=">2024-01-01",
        object_type="PAYLOAD",
        limit=5,
    )
    print(f"Found {len(recent)} records with epoch > 2024-01-01")

    # Less than: objects launched before a date
    old_sats = client.satcat(
        launch="<1970-01-01",
        current="Y",  # Still in orbit
        limit=5,
    )
    print(f"Found {len(old_sats)} satellites launched before 1970 still in orbit")

    # Not equal: exclude debris
    non_debris = client.gp(
        object_type="<>DEBRIS",
        limit=5,
    )
    print(f"Found {len(non_debris)} non-debris objects")
    # Found 5 records with epoch > 2024-01-01
    # Found 5 satellites launched before 1970 still in orbit
    # Found 5 non-debris objects

print("Example completed successfully!")
