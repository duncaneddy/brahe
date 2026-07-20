# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Constructing an UnscentedKalmanFilter using the builder API.

The builder takes the five required inputs -- epoch, state,
initial_covariance, force_config, and config -- directly as arguments to
builder(). Measurement models and remaining optional inputs are set
through chained setters.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

ukf = (
    bh.UnscentedKalmanFilter.builder(
        epoch, state, p0, bh.ForceModelConfig.two_body(), bh.UKFConfig()
    )
    .measurement_model(bh.InertialPositionMeasurementModel(10.0))
    .build()
)

print(f"UKF state dimension: {ukf.current_state().shape[0]}")
print(f"Records so far: {len(ukf.records())}")
