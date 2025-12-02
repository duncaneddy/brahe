# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates creating a SpaceTrack client with authentication.
"""

import os

import brahe as bh

# Get credentials from environment variables
# Note: For production use, most users should use the default constructor:
#   client = bh.SpaceTrackClient(user, password)
# These examples use the test server for automated testing.
user = os.environ.get("TEST_SPACETRACK_USER", "")
password = os.environ.get("TEST_SPACETRACK_PASS", "")

if not user or not password:
    print("Set TEST_SPACETRACK_USER and TEST_SPACETRACK_PASS environment variables")
    print("Register at: https://www.space-track.org/auth/createAccount")
else:
    # Create authenticated client using test server
    client = bh.SpaceTrackClient(
        user, password, base_url="https://for-testing-only.space-track.org/"
    )

    # Check authentication status
    print(f"Authenticated: {client.is_authenticated}")
    print(f"Base URL: {client.base_url}")
    # Authenticated: True
    # Base URL: https://for-testing-only.space-track.org/

print("Example completed successfully!")
