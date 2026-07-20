# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Constructing a NumericalOrbitPropagator using the builder API.

The builder takes the three required fields -- epoch, state, and
force_config -- directly as arguments to builder(). Optional fields such
as initial_covariance default to None when omitted and are set through
chained setters.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Minimal: only the three required fields
prop = bh.NumericalOrbitPropagator.builder(
    epoch, state, bh.ForceModelConfig.earth_gravity()
).build()

prop.propagate_to(epoch + 3600.0)
print(f"Minimal builder — epoch: {prop.current_epoch()}")

# With optional fields: custom propagation config and initial covariance
p0 = np.eye(6) * 1e6

prop_with_cov = (
    bh.NumericalOrbitPropagator.builder(
        epoch, state, bh.ForceModelConfig.earth_gravity()
    )
    .propagation_config(bh.NumericalPropagationConfig.high_precision())
    .initial_covariance(p0)
    .build()
)

prop_with_cov.propagate_to(epoch + 3600.0)

assert prop_with_cov.state_dim == 6
assert prop_with_cov.covariance(prop_with_cov.current_epoch()).shape == (6, 6)

print(f"Builder with covariance — epoch: {prop_with_cov.current_epoch()}")
print("Example validated successfully!")
