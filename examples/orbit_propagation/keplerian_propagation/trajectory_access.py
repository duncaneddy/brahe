# /// script
# dependencies = ["brahe"]
# ///
"""
Access and iterate over the KeplerianPropagator's internal trajectory
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)

# Propagate for several steps
prop.propagate_steps(5)

# Access trajectory
traj = prop.trajectory
print(f"Trajectory contains {len(traj)} states")
# Trajectory contains 6 states

# Iterate over epoch-state pairs
for epoch, state in traj:
    print(f"Epoch: {epoch}, semi-major axis: {state[0] / 1e3:.1f} km")
# Epoch: 2024-01-01 00:00:00.000 UTC, semi-major axis: 6878.1 km
# Epoch: 2024-01-01 00:01:00.000 UTC, semi-major axis: 6878.1 km
# Epoch: 2024-01-01 00:02:00.000 UTC, semi-major axis: 6878.1 km
# Epoch: 2024-01-01 00:03:00.000 UTC, semi-major axis: 6878.1 km
# Epoch: 2024-01-01 00:04:00.000 UTC, semi-major axis: 6878.1 km
# Epoch: 2024-01-01 00:05:00.000 UTC, semi-major axis: 6878.1 km
