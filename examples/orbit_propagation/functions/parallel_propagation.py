# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Parallel propagation of multiple satellites

This example demonstrates using par_propagate_to() to efficiently propagate
multiple satellites to a target epoch in parallel, useful for constellation
analysis and Monte Carlo simulations.
"""

import brahe as bh
import numpy as np
import time

bh.initialize_eop()

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Create multiple propagators for a constellation
num_sats = 10
propagators = []

for i in range(num_sats):
    # Vary semi-major axis slightly for each satellite
    a = bh.R_EARTH + 500e3 + i * 10e3
    oe = np.array([a, 0.001, 98.0, i * 36.0, 0.0, i * 36.0])
    state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
    propagators.append(prop)

# Target epoch: 24 hours later
target = epoch + 86400.0

# Propagate all satellites in parallel
start = time.time()
bh.par_propagate_to(propagators, target)
parallel_time = time.time() - start

print(f"Propagated {num_sats} satellites in parallel: {parallel_time:.4f} seconds")
print("\nFinal states:")
for i, prop in enumerate(propagators):
    state = prop.current_state()
    print(f"  Satellite {i}: r = {np.linalg.norm(state[:3]) / 1e3:.1f} km")
