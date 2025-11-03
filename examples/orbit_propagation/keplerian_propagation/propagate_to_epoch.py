# /// script
# dependencies = ["brahe"]
# ///
"""
Propagate KeplerianPropagator to a specific target epoch with precision
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)

# Propagate exactly 500 seconds (not evenly divisible by step size)
target = epoch + 500.0
prop.propagate_to(target)

print(f"Target epoch: {target}")
# Target epoch: 2024-01-01 00:08:20.000 UTC
print(f"Current epoch: {prop.current_epoch}")
# Current epoch: 2024-01-01 00:08:20.000 UTC
print(f"Difference: {abs(prop.current_epoch - target):.10f} seconds")
# Difference: 0.0000000000 seconds
# Output shows machine precision agreement
