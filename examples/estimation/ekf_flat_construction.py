# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Constructing an ExtendedKalmanFilter using the flat constructor.

The flat constructor takes every field as a keyword argument, as an
alternative to the builder when all inputs are already at hand.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

ekf = bh.ExtendedKalmanFilter(
    epoch,
    state,
    p0,
    measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
)

print(f"EKF state dimension: {ekf.current_state().shape[0]}")
print(f"EKF current epoch: {ekf.current_epoch()}")
