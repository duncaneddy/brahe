# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Enable the precomputed frame-rotation and ephemeris tables and compare against exact evaluation.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Fixed-step RK4 gives both propagations an identical step sequence
prop_config = bh.NumericalPropagationConfig(
    bh.IntegrationMethod.RK4,
    bh.IntegratorConfig.fixed_step(30.0),
    bh.VariationalConfig(),
)

gravity = bh.GravityConfiguration(degree=20, order=20)
third_body = [bh.ThirdBody.SUN, bh.ThirdBody.MOON]

# Exact rotation and DE440s ephemeris queries at every dynamics call (the default)
exact_config = bh.ForceModelConfig(gravity=gravity, third_body=third_body)

# Tabulate bias-precession-nutation and polar motion every 600 s and the
# Sun and Moon positions every 60 s
precomputed_config = bh.ForceModelConfig(
    gravity=gravity,
    third_body=third_body,
    precompute=bh.PrecomputeConfig(
        frame_rotation_step=600.0, ephemeris_step=60.0, cache_space_weather=False
    ),
)

exact = bh.NumericalOrbitPropagator(epoch, state, prop_config, exact_config, None)
precomputed = bh.NumericalOrbitPropagator(
    epoch, state, prop_config, precomputed_config, None
)

exact.propagate_to(epoch + 86400.0)
precomputed.propagate_to(epoch + 86400.0)

difference = np.linalg.norm(exact.current_state()[:3] - precomputed.current_state()[:3])
print(f"Position difference after one day: {difference:.2e} m")

assert difference < 1e-3
