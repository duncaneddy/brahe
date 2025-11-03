# /// script
# dependencies = ["brahe"]
# ///
"""
Step KeplerianPropagator forward one step at a time
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create propagator
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)

# Take one step (60 seconds)
prop.step()
print(f"After 1 step: {prop.current_epoch}")
# After 1 step: 2024-01-01 00:01:00.000 UTC

# Step by custom duration (120 seconds)
prop.step_by(120.0)
print(f"After custom step: {prop.current_epoch}")
# After custom step: 2024-01-01 00:03:00.000 UTC

# Trajectory now contains 3 states (initial + 2 steps)
print(f"Trajectory length: {len(prop.trajectory)}")
# Trajectory length: 3
