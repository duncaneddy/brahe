# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates querying GP (General Perturbations) data from SpaceTrack.
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

    # Query GP data for ISS (NORAD ID 25544)
    records = client.gp(norad_cat_id=25544, limit=1)

    # Records are returned as GPRecord objects with attribute access
    record = records[0]
    print(f"Object: {record.object_name}")
    print(f"NORAD ID: {record.norad_cat_id}")
    print(f"Epoch: {record.epoch}")
    print(f"Inclination: {record.inclination} deg")
    print(f"Eccentricity: {record.eccentricity}")
    # Object: ISS (ZARYA)
    # NORAD ID: 25544
    # Epoch: 2024-01-15 12:00:00
    # Inclination: 51.6416 deg
    # Eccentricity: 0.0006789

print("Example completed successfully!")
