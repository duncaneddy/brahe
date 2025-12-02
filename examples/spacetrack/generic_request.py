# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates making generic SpaceTrack API requests.
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

    # Make a generic request for custom queries
    # This returns raw response text
    response = client.generic_request(
        controller="basicspacedata",
        class_name="gp",
        predicates={"NORAD_CAT_ID": "25544", "limit": "1"},
    )

    print(f"Response length: {len(response)} characters")
    print(f"Response preview: {response[:100]}...")
    # Response length: 1234 characters
    # Response preview: [{"CCSDS_OMM_VERS":"2.0",...

print("Example completed successfully!")
