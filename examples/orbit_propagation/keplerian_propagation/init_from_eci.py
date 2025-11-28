# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize KeplerianPropagator from ECI Cartesian state vector
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Define Cartesian state in ECI frame [x, y, z, vx, vy, vz]
# Convert from Keplerian elements for this example
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state_eci = bh.state_koe_to_eci(elements, bh.AngleFormat.DEGREES)

# Create propagator from ECI state
prop = bh.KeplerianPropagator.from_eci(epoch, state_eci, 60.0)

print(f"Initial position magnitude: {np.linalg.norm(state_eci[:3]) / 1e3:.1f} km")
# Initial position magnitude: 6873.3 km
