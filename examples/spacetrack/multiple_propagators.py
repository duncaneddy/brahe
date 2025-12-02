# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates propagating multiple satellites from SpaceTrack data.
Uses the ICEYE constellation for reliable orbital data.
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

    # Fetch multiple satellites - use ICEYE constellation for reliable data
    gp_records = client.gp(
        object_name="~~ICEYE%",
        limit=5,
    )

    # Convert GP records to propagators using embedded TLE data
    propagators = []
    for record in gp_records:
        prop = bh.SGPPropagator.from_3le(
            record.object_name, record.tle_line1, record.tle_line2, step_size=60.0
        )
        propagators.append(prop)

    print(f"Retrieved {len(propagators)} propagators")

    # Propagate all to same epoch
    epoch = bh.Epoch.from_datetime(2024, 6, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    for i, prop in enumerate(propagators):
        prop.propagate_to(epoch)
        state = prop.current_state()
        r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
        altitude = (r - bh.R_EARTH) / 1000

        print(f"Satellite {i + 1}: altitude = {altitude:.1f} km")
    # Retrieved 5 propagators
    # Satellite 1: altitude = 570.5 km
    # Satellite 2: altitude = 568.2 km
    # ...

print("Example completed successfully!")
