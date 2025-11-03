# /// script
# dependencies = ["brahe"]
# ///
"""
Manage KeplerianPropagator trajectory memory with eviction policies
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)

# Keep only 100 most recent states
prop.set_eviction_policy_max_size(100)

# Propagate many steps
prop.propagate_steps(500)
print(f"Trajectory length: {len(prop.trajectory)}")  # Will be 100
# Trajectory length: 100

# Alternative: Keep only states within 1 hour of current time
prop.reset()
prop.set_eviction_policy_max_age(3600.0)  # 3600 seconds = 1 hour
prop.propagate_steps(500)
print(f"Trajectory length after age policy: {len(prop.trajectory)}")
# Trajectory length after age policy: 61
