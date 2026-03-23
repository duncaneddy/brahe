# /// script
# dependencies = ["brahe"]
# ///
"""
Change KeplerianPropagator step size during propagation
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)

print(f"Initial step size: {prop.step_size} seconds")

# Change step size
prop.set_step_size(120.0)
print(f"New step size: {prop.step_size} seconds")

# Subsequent steps use new step size
prop.step()  # Advances 120 seconds
print(f"After step: {(prop.current_epoch() - epoch):.1f} seconds elapsed")
