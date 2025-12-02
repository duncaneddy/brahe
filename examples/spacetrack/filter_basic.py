# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates basic filtering in SpaceTrack queries.
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

    # Filter by object type
    payloads = client.satcat(
        object_type="PAYLOAD",
        country="US",
        limit=5,
    )
    print(f"Found {len(payloads)} US payloads")

    # Filter by country code in GP data
    gp_records = client.gp(
        country_code="US",
        object_type="PAYLOAD",
        limit=5,
    )
    print(f"Found {len(gp_records)} GP records for US payloads")
    # Found 5 US payloads
    # Found 5 GP records for US payloads

print("Example completed successfully!")
