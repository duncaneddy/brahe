# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Permanent tide correction + static + time-varying solid Earth tide.
Enables both the IERS static (frequency-independent) and time-varying (frequency-dependent) corrections.
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

# Permanent + static + time-varying solid tide: static (frequency-independent) correction
# always on, plus time-varying frequency-dependent refinements from Tables 6.5a/b/c.
# Recommended for precise orbit determination and geodesy applications.
solid = bh.SolidTideConfig(frequency_dependent=True)
tides = bh.TidesConfiguration(permanent=bh.PermanentTideConfig.AUTO, solid=solid)

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
    f"Static + time-varying solid-tide example: position after 60 s = "
    f"({state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}) m"
)
print("Example validated successfully!")
