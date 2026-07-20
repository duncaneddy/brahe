# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Constructing a BatchLeastSquares estimator using the builder API.

The builder takes the five required inputs -- epoch, state,
apriori_covariance, force_config, and config -- directly as arguments to
builder(). Measurement models and remaining optional inputs are set
through chained setters.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

bls = (
    bh.BatchLeastSquares.builder(
        epoch, state, p0, bh.ForceModelConfig.two_body(), bh.BLSConfig()
    )
    .measurement_model(bh.InertialPositionMeasurementModel(10.0))
    .build()
)

print(f"BLS state dimension: {bls.current_state().shape[0]}")
print(f"Converged: {bls.converged()}")
