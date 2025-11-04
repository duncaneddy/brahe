# /// script
# dependencies = ["brahe"]
# ///
"""
Track KeplerianPropagator with names and IDs for multi-satellite scenarios
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])

# Create propagator with identity (builder pattern)
prop = (
    bh.KeplerianPropagator.from_keplerian(epoch, elements, bh.AngleFormat.DEGREES, 60.0)
    .with_name("Satellite-A")
    .with_id(12345)
)

print(f"Name: {prop.get_name()}")
# Name: Satellite-A
print(f"ID: {prop.get_id()}")
# ID: 12345
print(f"UUID: {prop.get_uuid()}")
# UUID: None (because not set)
