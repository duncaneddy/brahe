# /// script
# dependencies = ["brahe"]
# ///
"""
Propagate KeplerianPropagator forward multiple steps at once
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)

# Take 10 steps (10 × 60 = 600 seconds)
prop.propagate_steps(10)
print(f"After 10 steps: {(prop.current_epoch - epoch):.1f} seconds elapsed")
# After 10 steps: 600.0 seconds elapsed
print(f"Trajectory length: {len(prop.trajectory)}")
# Trajectory length: 11
