# /// script
# dependencies = ["brahe"]
# ///
"""
Parse an AEM and convert a segment to an AttitudeTrajectory for interpolation.
"""

import brahe as bh
from brahe.ccsds import AEM

bh.initialize_eop()

# Parse an AEM with two quaternion segments
aem = AEM.from_file("test_assets/ccsds/aem/AEMExampleG4.txt")

# Segment 1 carries no INTERPOLATION_METHOD, so it converts cleanly to the
# default slerp trajectory. Segment 0 sets INTERPOLATION_METHOD = HERMITE,
# which has no AttitudeTrajectory equivalent and would raise an error.
traj = aem.segment_to_attitude_trajectory(1)
print(f"Trajectory: {len(traj)} states")
print(f"  Frame A:       {traj.frame_a}")
print(f"  Frame B:       {traj.frame_b}")
print(f"  Interpolation: {traj.interpolation_method}")
print(f"  Has rates:     {traj.has_rates}")
print(f"  Start:         {traj.start_epoch}")
print(f"  End:           {traj.end_epoch}")

# Slerp-query the attitude at the midpoint of the trajectory's span
t0 = traj.start_epoch
t1 = traj.end_epoch
mid = t0 + (t1 - t0) / 2.0
quaternion = traj.quaternion(mid)
wire = quaternion.to_vector(scalar_first=False)
print(
    f"\nInterpolated quaternion [Q1, Q2, Q3, QC] at {mid}: "
    f"[{wire[0]:.5f}, {wire[1]:.5f}, {wire[2]:.5f}, {wire[3]:.5f}]"
)
