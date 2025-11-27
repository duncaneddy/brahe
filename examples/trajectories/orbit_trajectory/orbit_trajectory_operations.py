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
    state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
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

# Output:
# Length: 10
# Timespan: 540.0 seconds
# Start epoch: 2024-01-01 00:00:00.000 UTC
# End epoch: 2024-01-01 00:09:00.000 UTC

# Interpolated state at 2024-01-01 00:00:45.000 UTC:
#   Position (km): [1159.01597302 6101.29789026 2925.16369358]
#   Velocity (m/s): [-5578.86734152 -1338.77483001  5004.22925364]
# State 0: Epoch=2024-01-01 00:00:00.000 UTC, Position magnitude=6871.26 km
# State 1: Epoch=2024-01-01 00:01:00.000 UTC, Position magnitude=6871.29 km
