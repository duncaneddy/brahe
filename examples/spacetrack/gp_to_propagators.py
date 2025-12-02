# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates converting GP data to SGP propagators.
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

    # Fetch GP data and convert to propagators in one step
    propagators = client.gp_as_propagators(
        step_size=60.0,  # 60 second propagation step
        norad_cat_id=25544,  # ISS
        limit=1,
    )

    prop = propagators[0]

    # Propagate to a specific epoch
    epoch = bh.Epoch.from_datetime(2024, 6, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    prop.propagate_to(epoch)
    state = prop.current_state()

    # State is [x, y, z, vx, vy, vz] in meters and m/s
    r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
    altitude = r - bh.R_EARTH

    print(f"ISS position at {epoch}:")
    print(f"  Altitude: {altitude / 1000:.1f} km")
    print(
        f"  Position: [{state[0] / 1000:.1f}, {state[1] / 1000:.1f}, {state[2] / 1000:.1f}] km"
    )
    # ISS position at 2024-06-01T12:00:00.000000000Z UTC:
    #   Altitude: 420.5 km
    #   Position: [1234.5, 5678.9, 2345.6] km

print("Example completed successfully!")
