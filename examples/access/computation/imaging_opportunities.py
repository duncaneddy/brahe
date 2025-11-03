# /// script
# dependencies = ["brahe"]
# ///
"""
Find imaging opportunities using polygon locations and complex constraint combinations
"""

import brahe as bh

bh.initialize_eop()

# Define imaging target (polygon region of interest - SF Bay Area)
target_vertices = [
    [-122.5, 37.7, 0.0],
    [-122.3, 37.7, 0.0],
    [-122.3, 37.9, 0.0],
    [-122.5, 37.9, 0.0],
    [-122.5, 37.7, 0.0],
]
target = bh.PolygonLocation(target_vertices).with_name("SF Bay Area")

# Imaging satellite (using ISS as example)
tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2, 60.0).with_name("EO-Sat-1")

# Complex constraint: daylight + off-nadir < 30° + right-looking
daylight = bh.LocalTimeConstraint(time_windows=[(800, 1800)])
off_nadir = bh.OffNadirConstraint(min_off_nadir_deg=0.0, max_off_nadir_deg=30.0)
look_right = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)

# Combine constraints with AND logic
constraint = bh.ConstraintAll(constraints=[daylight, off_nadir, look_right])

# Configure for imaging (shorter time steps for accuracy)
config = bh.AccessSearchConfig(
    initial_time_step=30.0,
    adaptive_step=True,
    adaptive_fraction=0.05,
    parallel=False,
    num_threads=0,
)

# Compute imaging opportunities over next 10 days
epoch_start = bh.Epoch(2008, 9, 20, 0, 0, 0.0, 0.0)
epoch_end = epoch_start + 10 * 86400.0

windows = bh.location_accesses(
    target, propagator, epoch_start, epoch_end, constraint, config=config
)

print(f"Found {len(windows)} imaging opportunities")
for i, window in enumerate(windows[:3], 1):
    print(f"\nOpportunity {i}:")
    print(f"  Start: {window.window_open}")
    print(f"  Duration: {window.duration / 60:.1f} min")

    # Access imaging-specific properties
    off_nadir_min = window.properties.off_nadir_min
    print(f"  Off-nadir: {off_nadir_min:.1f}°")

    local_time = window.properties.local_time
    hours = int(local_time)
    minutes = int((local_time - hours) * 60)
    print(f"  Local time: {hours:02d}:{minutes:02d}")

# Expected output (values will vary):
# Found X imaging opportunities
#
# Opportunity 1:
#   Start: 2008-09-XX HH:MM:SS.SSS UTC
#   Duration: X.X min
#   Off-nadir: XX.X°
#   Local time: HH:MM
