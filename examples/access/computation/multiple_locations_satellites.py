# /// script
# dependencies = ["brahe"]
# ///
"""
Compute access windows for multiple ground locations and multiple satellites
"""

import brahe as bh
from collections import defaultdict

bh.initialize_eop()

# Define multiple ground stations
locations = [
    bh.PointLocation(-122.4194, 37.7749, 0.0).with_name("San Francisco"),
    bh.PointLocation(-71.0589, 42.3601, 0.0).with_name("Boston"),
    bh.PointLocation(15.4038, 78.2232, 458.0).with_name("Svalbard"),
]

# Define multiple satellites (from TLEs, epoch: 2024-01-01)
tle_data = [
    # ISS - LEO, 51.6° inclination
    (
        "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999",
        "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601",
        "ISS",
    ),
    # Tiangong - LEO, 41.5° inclination
    (
        "1 48274U 21035A   25306.17586037  .00031797  00000-0  38131-3 0  9995",
        "2 48274  41.4666 263.0710 0006682 308.7013  51.3228 15.60215133257694",
        "Tiangong",
    ),
]

propagators = [
    bh.SGPPropagator.from_tle(line1, line2, 60.0).with_name(name)
    for line1, line2, name in tle_data
]

# Compute all location-satellite pairs (24 hours from TLE epoch)
epoch_start = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0)
epoch_end = epoch_start + 86400.0  # 24 hours
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

windows = bh.location_accesses(
    locations, propagators, epoch_start, epoch_end, constraint
)

# Results include windows for all location-satellite combinations
print(f"Total windows: {len(windows)}")

# Group by location
by_location = defaultdict(list)
for window in windows:
    by_location[window.location_name].append(window)

for loc_name, loc_windows in by_location.items():
    print(f"\n{loc_name}: {len(loc_windows)} windows")

# Output:
# Total windows: 20

# Boston: 10 windows

# San Francisco: 10 windows
