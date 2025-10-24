# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
This example shows how to find passes of the ISS over San Francisco, CA
using an elevation constraint.
"""

import brahe as bh

# Initialize EOP
bh.initialize_eop()

# Set the location
location = bh.PointLocation(-122.4194, 37.7749, 0.0).with_name("San Francisco")

# Get the latest TLE for the ISS (NORAD ID 25544) from Celestrak
propagator = bh.datasets.celestrak.get_tle_by_id_as_propagator(25544, 60.0)

# Configure Search Window
epoch_start = bh.Epoch.now()
epoch_end = epoch_start + 7 * 86400.0  # 7 days later

# Set access constraints -> Must be above 10 degrees elevation
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

# Compute access windows
windows = bh.location_accesses(location, propagator, epoch_start, epoch_end, constraint)

assert len(windows) > 0, "Should find at least one access window"

# Print first 3 access windows
for window in windows[:3]:
    print(
        f"Access Window: {window.window_open} to {window.window_close}, Duration: {window.duration / 60:.2f} minutes"
    )
# Outputs:
# Access Window: 2025-10-25 08:49:40.062 UTC to 2025-10-25 08:53:48.463 UTC, Duration: 4.14 minutes
# Access Window: 2025-10-25 10:25:40.245 UTC to 2025-10-25 10:31:48.463 UTC, Duration: 6.14 minutes
# Access Window: 2025-10-25 12:05:33.455 UTC to 2025-10-25 12:06:48.463 UTC, Duration: 1.25 minutes
