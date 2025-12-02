# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates querying satellite constellations from SpaceTrack.
Uses the ICEYE constellation as a smaller example.
"""

import os

import numpy as np

import brahe as bh

# Initialize EOP data for propagation
bh.initialize_eop()

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

    # Query ICEYE constellation (smaller than Starlink, good for demo)
    # First query GP records with pattern matching
    iceye_records = client.gp(
        object_name="~~ICEYE%",
        limit=10,
    )

    # Convert GP records to propagators using embedded TLE data
    iceye_props = []
    for r in iceye_records:
        prop = bh.SGPPropagator.from_3le(
            r.object_name, r.tle_line1, r.tle_line2, step_size=60.0
        )
        iceye_props.append(prop)

    print(f"Retrieved {len(iceye_props)} ICEYE propagators")

    # Calculate orbital statistics
    epoch = bh.Epoch.from_datetime(2024, 6, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    altitudes = []

    for prop in iceye_props:
        prop.propagate_to(epoch)
        state = prop.current_state()
        r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
        altitudes.append((r - bh.R_EARTH) / 1000)

    print(f"Altitude range: {min(altitudes):.1f} - {max(altitudes):.1f} km")
    print(f"Mean altitude: {np.mean(altitudes):.1f} km")
    # Retrieved 10 ICEYE propagators
    # Altitude range: 560.2 - 580.8 km
    # Mean altitude: 570.3 km

print("Example completed successfully!")
