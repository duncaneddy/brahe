# /// script
# dependencies = ["brahe"]
# ///
"""
Parse an OEM and convert segments to OrbitTrajectory objects for interpolation.
"""

import brahe as bh
from brahe.ccsds import OEM

bh.initialize_eop()

# Parse an OEM file
oem = OEM.from_file("test_assets/ccsds/oem/OEMExample5.txt")
seg = oem.segments[0]
print(f"Segment: {seg.object_name}, {seg.num_states} states, frame={seg.ref_frame}")
# Segment: ISS, 49 states, frame=GCRF

# Convert segment 0 to an OrbitTrajectory
traj = oem.segment_to_orbit_trajectory(0)
print(f"\nTrajectory: {len(traj)} states")
print(f"  Frame: {traj.frame}")
print(f"  Start: {traj.start_epoch()}")
print(f"  End:   {traj.end_epoch()}")
print(f"  Span:  {traj.timespan():.0f} seconds")
# Trajectory: 49 states
#   Frame: ...
#   Start: ...
#   End:   ...
#   Span:  ... seconds

# Access states by index
epc, state = traj.get(0)
print("\nFirst state:")
print(f"  Epoch: {epc}")
print(
    f"  Position: [{state[0] / 1e3:.3f}, {state[1] / 1e3:.3f}, {state[2] / 1e3:.3f}] km"
)
print(f"  Velocity: [{state[3]:.3f}, {state[4]:.3f}, {state[5]:.3f}] m/s")

# Interpolate at an arbitrary epoch between states
epc0, _ = traj.get(0)
epc1, _ = traj.get(1)
mid_epoch = epc0 + (epc1 - epc0) / 2.0
interp_state = traj.interpolate(mid_epoch)
print(f"\nInterpolated state at {mid_epoch}:")
print(
    f"  Position: [{interp_state[0] / 1e3:.3f}, {interp_state[1] / 1e3:.3f}, {interp_state[2] / 1e3:.3f}] km"
)

# Convert all segments at once
oem_multi = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
trajs = oem_multi.to_orbit_trajectories()
print(f"\nMulti-segment OEM: {len(trajs)} trajectories")
for i, t in enumerate(trajs):
    print(f"  [{i}] {len(t)} states, span={t.timespan():.0f}s")
