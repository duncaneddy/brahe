# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Permanent-tide correction only (no solid Earth tides).
Corrects the geopotential model's C̄20 for its tide system but adds no
time-varying solid-tide accelerations.
"""

import numpy as np
import brahe as bh

# EOP is required for the ITRF frame transformations used inside the tidal model.
bh.initialize_eop()
bh.initialize_sw()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# LEO satellite: 500 km altitude, slightly elliptical, sun-synchronous inclination.
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state0 = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Permanent tide only: corrects C̄20 for the tide system of the loaded model
# but adds no time-varying solid-tide accelerations.
tides = bh.TidesConfiguration(permanent=bh.PermanentTideConfig.AUTO, solid=None)

force_config = bh.ForceModelConfig.earth_gravity()
force_config.tides = tides

prop = bh.NumericalOrbitPropagator(
    epoch,
    state0.copy(),
    bh.NumericalPropagationConfig.default(),
    force_config,
)

t1 = epoch + 60.0  # step once by 60 seconds
prop.propagate_to(t1)
state = prop.current_state()
print(
    f"Permanent-tide-only example: position after 60 s = "
    f"({state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}) m"
)
print("Example validated successfully!")
