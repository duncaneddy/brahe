# /// script
# dependencies = ["brahe"]
# ///
"""
Query KeplerianPropagator states at multiple epochs in batch
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)

# Generate states at irregular intervals
query_epochs = [epoch + t for t in [0.0, 100.0, 500.0, 1000.0, 3600.0]]
states_eci = prop.states_eci(query_epochs)

print(f"Generated {len(states_eci)} states")
for i, state in enumerate(states_eci):
    print(f"  Epoch {i}: position magnitude = {np.linalg.norm(state[:3]) / 1e3:.1f} km")
