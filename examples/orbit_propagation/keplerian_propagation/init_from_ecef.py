# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize KeplerianPropagator from ECEF Cartesian state vector
"""

import brahe as bh
import numpy as np

bh.initialize_eop()  # Required for ECEF ↔ ECI transformations

epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Get state in ECI, then convert to ECEF for demonstration
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state_eci = bh.state_koe_to_eci(elements, bh.AngleFormat.DEGREES)
state_ecef = bh.state_eci_to_ecef(epoch, state_eci)

# Create propagator from ECEF state
prop = bh.KeplerianPropagator.from_ecef(epoch, state_ecef, 60.0)

print(f"ECEF position magnitude: {np.linalg.norm(state_ecef[:3]) / 1e3:.1f} km")
# ECEF position magnitude: 6873.3 km
