# /// script
# dependencies = ["brahe"]
# ///
"""
Query KeplerianPropagator state at arbitrary epochs without building trajectory
"""

import brahe as bh
import numpy as np

bh.initialize_eop()  # Required for frame transformations

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)

# Query state 1 hour later (doesn't add to trajectory)
query_epoch = epoch + 3600.0
state_native = prop.state(
    query_epoch
)  # Native format of propagator internal state (Keplerian)
state_eci = prop.state_eci(query_epoch)  # ECI Cartesian
state_ecef = prop.state_ecef(query_epoch)  # ECEF Cartesian
state_kep = prop.state_as_osculating_elements(query_epoch, bh.AngleFormat.DEGREES)

print(f"Native state (Keplerian): a={state_native[0] / 1e3:.1f} km")
# Native state (Keplerian): a=6878.1 km
print(f"ECI position magnitude: {np.linalg.norm(state_eci[:3]) / 1e3:.1f} km")
# ECI position magnitude: 6877.7 km
print(f"ECEF position magnitude: {np.linalg.norm(state_ecef[:3]) / 1e3:.1f} km")
# ECEF position magnitude: 6877.7 km
