# /// script
# dependencies = ["brahe"]
# ///
"""
Download a Starlink satellite's ephemeris, interpolate its trajectory, read
its covariance in another frame, and compute ground-station access windows.

Uses the manifest and ephemeris cached under the brahe cache directory
(seeded by `just seed-starlink-cache`).
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

client = bh.StarlinkClient(cache_max_age=7 * 86400.0)
manifest = client.get_manifest()
entry = manifest.find_by_object_name("STARLINK-38128")

trajectory = client.get_trajectory(entry.norad_cat_id)
start = trajectory.start_epoch()
end = trajectory.end_epoch()
print(
    f"Trajectory: {len(trajectory)} samples from {start} to {end} in {trajectory.frame}"
)

mid = start + 30.0
state = trajectory.interpolate(mid)
print(f"State at {mid} [m, m/s]: {np.array2string(state, precision=3)}")

covariance = trajectory.covariance_in_frame(bh.CelestialFrame.ITRF, mid)
print(
    f"ITRF position 1-sigma at {mid} [m]: "
    f"{np.array2string(np.sqrt(np.diag(covariance))[:3], precision=3)}"
)

station = bh.PointLocation(-122.4194, 37.7749, 0.0).with_name("San Francisco")
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
windows = bh.location_accesses(station, trajectory, start, end, constraint)
print(f"Access windows above 10 deg: {len(windows)}")
for window in windows[:2]:
    print(f"  {window.window_open} to {window.window_close} ({window.duration:.1f} s)")
