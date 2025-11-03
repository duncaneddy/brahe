# /// script
# dependencies = ["brahe"]
# ///
"""
Reset KeplerianPropagator to initial conditions
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)

# Propagate forward
prop.propagate_steps(100)
print(f"After propagation: {len(prop.trajectory)} states")
# After propagation: 101 states

# Reset to initial conditions
prop.reset()
print(f"After reset: {len(prop.trajectory)} states")
# After reset: 1 states
print(f"Current epoch: {prop.current_epoch}")
# Current epoch: 2024-01-01 00:00:00.000 UTC
