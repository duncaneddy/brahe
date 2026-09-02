# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Query an AttitudeTrajectory through the OrientationProvider accessors.
"""

import math

import numpy as np

import brahe as bh

bh.initialize_eop()

# Two samples 60 seconds apart, rotating about the spacecraft Z axis at
# 0.01 rad/s. The rate is carried on both states, so angular_velocity is
# available alongside the attitude.
traj = bh.AttitudeTrajectory(
    bh.ReferenceFrame.celestial(bh.CelestialFrame.GCRF),
    bh.ReferenceFrame.body(None, bh.BodyFrame.SC_BODY("1")),
)

omega = np.array([0.0, 0.0, 0.01])
t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
t1 = t0 + 60.0
half_angle = 0.5 * omega[2] * 60.0
traj.add(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0), omega)
traj.add(t1, bh.Quaternion(math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)), omega)

# Every accessor evaluates the same interpolated attitude and returns it in a
# different representation.
epoch = t0 + 30.0
quaternion = traj.quaternion(epoch)
angles = traj.euler_angle(epoch, bh.EulerAngleOrder.ZYX)
axis = traj.euler_axis(epoch)
matrix = traj.rotation_matrix(epoch).to_matrix()
rate = traj.angular_velocity(epoch)

q = quaternion.to_vector(scalar_first=True)
print(f"Coverage:    {traj.start_epoch} to {traj.end_epoch}")
print(f"Query epoch: {epoch}")
print(f"Quaternion:  s {q[0]:.6f}, v {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
print(
    "Euler ZYX:   "
    f"{math.degrees(angles.phi):.4f}, "
    f"{math.degrees(angles.theta):.4f}, "
    f"{math.degrees(angles.psi):.4f} deg"
)
print(
    "Euler axis:  "
    f"{axis.axis[0]:.4f} {axis.axis[1]:.4f} {axis.axis[2]:.4f}, "
    f"angle {math.degrees(axis.angle):.4f} deg"
)
print("Rotation matrix:")
for row in matrix:
    print(f"  {row[0]:9.6f} {row[1]:9.6f} {row[2]:9.6f}")
print(f"Rate:        {rate[0]:.4f} {rate[1]:.4f} {rate[2]:.4f} rad/s")
