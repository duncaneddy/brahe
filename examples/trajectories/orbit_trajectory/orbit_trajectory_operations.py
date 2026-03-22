# /// script
# dependencies = ["brahe"]
# ///
"""
Use standard trajectory operations (length, timespan, interpolation)
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory
traj = bh.OrbitTrajectory(6, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)

# Add states
epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
for i in range(10):
    epoch = epoch0 + i * 60.0
    oe = np.array([bh.R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, i * 0.1])
    state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    traj.add(epoch, state)

# Query properties
print(f"Length: {len(traj)}")
print(f"Timespan: {traj.timespan():.1f} seconds")
print(f"Start epoch: {traj.start_epoch()}")
print(f"End epoch: {traj.end_epoch()}")

# Interpolate at intermediate time
interp_epoch = epoch0 + 45.0
interp_state = traj.interpolate(interp_epoch)
print(f"\nInterpolated state at {interp_epoch}:")
print(f"  Position (km): {interp_state[0:3] / 1e3}")
print(f"  Velocity (m/s): {interp_state[3:6]}")

# Iterate over states
for i, (epoch, state) in enumerate(traj):
    if i < 2:  # Just show first two
        print(
            f"State {i}: Epoch={epoch}, Position magnitude={np.linalg.norm(state[0:3]) / 1e3:.2f} km"
        )
