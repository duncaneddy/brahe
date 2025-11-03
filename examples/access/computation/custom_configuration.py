# /// script
# dependencies = ["brahe"]
# ///
"""
Configure access search algorithm with custom parameters for performance tuning
"""

import brahe as bh

bh.initialize_eop()

# Create custom configuration
config = bh.AccessSearchConfig(
    initial_time_step=60.0,  # Coarse search: 60-second steps
    adaptive_step=True,  # Enable adaptive refinement
    adaptive_fraction=0.75,  # Each step is 75% of orbital period
    parallel=True,  # Enable parallel processing
    num_threads=0,  # Auto-detect thread count
)

# Use custom config with location and propagator
location = bh.PointLocation(-122.4194, 37.7749, 0.0).with_name("San Francisco")
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2, 60.0).with_name("ISS")

epoch_start = bh.Epoch(2025, 11, 2, 0, 0, 0.0, 0.0)
epoch_end = epoch_start + 86400.0  # 24 hours
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

windows = bh.location_accesses(
    location, propagator, epoch_start, epoch_end, constraint, config=config
)

print(f"Found {len(windows)} access windows with custom configuration")
print(
    f"Configuration: {config.initial_time_step}s time step, adaptive={config.adaptive_step}"
)

# Expected output:
# Found 5 access windows with custom configuration
# Configuration: 60s time step, adaptive=True
