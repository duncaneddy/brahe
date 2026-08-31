# /// script
# dependencies = ["brahe"]
# ///
"""
Build an AttitudeTrajectory in code and compare Slerp and Linear interpolation.
"""

import math

import brahe as bh

bh.initialize_eop()

# Two attitude samples 60 seconds apart: a constant-rate rotation of 2 deg/s
# about the spacecraft Z axis, from 0 to 120 degrees.
traj = bh.AttitudeTrajectory(
    bh.AttitudeFrame.reference_frame(bh.ReferenceFrame.GCRF),
    bh.AttitudeFrame.spacecraft_body_frame("SC_BODY"),
)

t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
t1 = t0 + 60.0
q0 = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
half_angle = math.radians(60.0)
q1 = bh.Quaternion(math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))
traj.add(t0, q0)
traj.add(t1, q1)

# Query one third of the way between the two nodes. At the exact midpoint,
# linear interpolation of a single-axis rotation happens to coincide with
# slerp, so an off-center query is needed to see them diverge.
query = t0 + 20.0

traj.set_interpolation_method("SLERP")
slerp_q = traj.quaternion(query)

traj.set_interpolation_method("LINEAR")
linear_q = traj.quaternion(query)

slerp_angle_deg = math.degrees(slerp_q.to_euler_axis().angle)
linear_angle_deg = math.degrees(linear_q.to_euler_axis().angle)

print(f"Query epoch: {query} (1/3 of the way from t0 to t1)")
print(f"Slerp  rotation angle:  {slerp_angle_deg:.4f} deg (exact)")
print(f"Linear rotation angle:  {linear_angle_deg:.4f} deg (approximate)")
