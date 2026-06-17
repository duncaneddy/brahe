"""Force model configuration with permanent tide + static and time-varying solid Earth tides."""

import brahe as bh

# Permanent tide + both the static and time-varying (frequency-dependent) parts
# of the solid Earth tide correction. frequency_dependent=True adds the
# tidal-line refinements (IERS Tables 6.5a/b/c) on top of the static correction.
tides = bh.TidesConfiguration(
    permanent=bh.PermanentTideConfig.AUTO,
    solid=bh.SolidTideConfig(frequency_dependent=True),
)

force_config = bh.ForceModelConfig.earth_gravity()
force_config.tides = tides

# `force_config` is now ready to hand to a NumericalOrbitPropagator.
assert force_config.tides is not None
