"""Force model configuration with the permanent-tide correction only."""

import brahe as bh

# Permanent-tide correction only: normalize the loaded model's C̄20 to the
# conventional tide-free system, but add no time-varying solid-tide accelerations.
tides = bh.TidesConfiguration(
    permanent=bh.PermanentTideConfig.AUTO,
    solid=None,
)

force_config = bh.ForceModelConfig.earth_gravity()
force_config.tides = tides

# `force_config` is now ready to hand to a NumericalOrbitPropagator.
assert force_config.tides is not None
