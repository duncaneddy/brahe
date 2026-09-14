# /// script
# dependencies = ["brahe"]
# ///
"""
Convert a Modified ITC ephemeris to a trajectory and use it.

Builds an OrbitTrajectory in the file's EME2000 frame with the RTN
covariance rotated into that frame, interpolates a state between two
records, converts the trajectory to ITRF, and computes ground-station
access windows over the ephemeris span.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

PATH = "test_assets/starlink/MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt"

itc = bh.ITC.from_file(PATH)
trajectory = itc.to_trajectory()
start = trajectory.start_epoch()
end = trajectory.end_epoch()
print(f"Trajectory: {len(trajectory)} samples from {start} to {end}")

mid = start + 30.0
state = trajectory.interpolate(mid)
print(f"State at {mid} [m, m/s]: {np.array2string(state, precision=3)}")
covariance = trajectory.covariance(mid)
print(
    f"Position 1-sigma at {mid} [m]: {np.array2string(np.sqrt(np.diag(covariance))[:3], precision=3)}"
)

itrf = trajectory.to_itrf()
ecef = itrf.interpolate(mid)
print(f"ITRF position at {mid} [m]: {np.array2string(ecef[:3], precision=3)}")

station = bh.PointLocation(-122.4194, 37.7749, 0.0).with_name("San Francisco")
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
windows = bh.location_accesses(station, trajectory, start, end, constraint)
print(f"Access windows above 10 deg: {len(windows)}")
for window in windows[:3]:
    print(f"  {window.window_open} to {window.window_close} ({window.duration:.1f} s)")
