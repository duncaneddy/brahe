# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build an AttitudeTrajectory and show the uniform rate-presence rule that add enforces.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

# A trajectory relates two frame endpoints: every stored quaternion rotates
# from frame_a into frame_b.
traj = bh.AttitudeTrajectory(
    bh.ReferenceFrame.celestial(bh.CelestialFrame.GCRF),
    bh.ReferenceFrame.body(None, bh.BodyFrame.SC_BODY("1")),
)

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
traj.add(epoch, bh.Quaternion(1.0, 0.0, 0.0, 0.0))

print(f"frame_a:   {traj.frame_a}")
print(f"frame_b:   {traj.frame_b}")
print(f"States:    {len(traj)}")
print(f"has_rates: {traj.has_rates}")

# The first state carries no angular velocity, so every later state must omit
# it as well. Adding a rate-carrying state to a rate-free trajectory is
# rejected rather than producing a mixed trajectory.
try:
    traj.add(
        epoch + 60.0,
        bh.Quaternion(1.0, 0.0, 0.0, 0.0),
        np.array([0.0, 0.0, 0.01]),
    )
except bh.BraheError as exc:
    print(f"\nMixed rate rejected: {exc}")
