"""Force model configuration with permanent tide + the static solid Earth tide."""

import brahe as bh

# Permanent tide + the static (frequency-independent) solid Earth tide correction.
# frequency_dependent=False keeps only the always-on static part.
tides = bh.TidesConfiguration(
    permanent=bh.PermanentTideConfig.AUTO,
    solid=bh.SolidTideConfig(frequency_dependent=False),
)

force_config = bh.ForceModelConfig.earth_gravity()
force_config.tides = tides

# `force_config` is now ready to hand to a NumericalOrbitPropagator.
assert force_config.tides is not None
