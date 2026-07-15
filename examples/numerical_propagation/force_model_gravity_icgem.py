# /// script
# dependencies = ["brahe", "numpy"]
# FLAGS = ["NETWORK"]
# ///
"""
Configure a NumericalOrbitPropagator with an ICGEM-sourced gravity model.

GravityModelType.icgem(body, name) slots into the same model_type slot as
the packaged JGM3/GGM05S/EGM2008_120 constants. The .gfc file is downloaded
into $BRAHE_CACHE/icgem/ on first use of the resulting GravityModel.
"""

import brahe as bh
import numpy as np

# Initialize EOP data (required for any numerical propagation)
bh.initialize_eop()

# Reference an ICGEM Earth model. Use bh.datasets.icgem.list_models("earth")
# to discover the full catalog. Append "-<DEGREE>" to pin a specific variant.
grav_type = bh.GravityModelType.icgem("earth", "JGM3")

gravity_cfg = bh.GravityConfiguration.spherical_harmonic(
    degree=20, order=20, model_type=grav_type
)

# Minimal force model: ICGEM-sourced spherical-harmonic gravity only
force_cfg = bh.ForceModelConfig(gravity=gravity_cfg)

# Build an initial state for a LEO satellite
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array(
    [
        bh.R_EARTH + 500e3,  # a (m)
        0.001,  # e
        np.radians(97.8),  # i (rad)
        np.radians(15.0),  # RAAN (rad)
        np.radians(30.0),  # arg perigee (rad)
        np.radians(45.0),  # true anomaly (rad)
    ]
)
state0 = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)

# Construct the propagator — this is where the ICGEM model is downloaded
# (if not cached) and loaded into the force evaluator.
prop = bh.NumericalOrbitPropagator(
    epoch,
    state0,
    bh.NumericalPropagationConfig.default(),
    force_cfg,
    None,
)

# Step one minute forward
prop.step_by(60.0)
state1 = prop.current_state()

drift = float(np.linalg.norm(np.asarray(state1[:3]) - np.asarray(state0[:3])))
print(f"Propagated 60 s with JGM3 (ICGEM source); position drift = {drift:.1f} m")
