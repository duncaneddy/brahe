# /// script
# dependencies = ["brahe"]
# ///
"""
Basic access computation workflow: finding satellite passes over a ground location
"""

import brahe as bh

# Initialize Earth orientation data
bh.initialize_eop()

# Define ground location (San Francisco, CA)
location = bh.PointLocation(-122.4194, 37.7749, 0.0).with_name("San Francisco")

# Create propagator from TLE (example for ISS)
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2, 60.0).with_name("ISS")

# Define time window (7 days starting from epoch)
epoch_start = bh.Epoch(2025, 11, 2, 0, 0, 0.0, 0.0)
epoch_end = epoch_start + 7 * 86400.0

# Define constraint (minimum 10° elevation)
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

# Compute access windows
windows = bh.location_accesses(location, propagator, epoch_start, epoch_end, constraint)

# Process results
print(f"Found {len(windows)} access windows")
for i, window in enumerate(windows[:3], 1):
    duration_min = window.duration / 60.0
    print(f"\nWindow {i}:")
    print(f"  Start: {window.window_open}")
    print(f"  End:   {window.window_close}")
    print(f"  Duration: {duration_min:.2f} minutes")

    # Access computed properties
    elev_max = window.properties.elevation_max
    print(f"  Max elevation: {elev_max:.1f}°")
