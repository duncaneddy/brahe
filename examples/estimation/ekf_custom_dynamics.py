# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Create an EKF with custom dynamics using additional_dynamics parameter.

Demonstrates how to add custom two-body acceleration as additional dynamics
to the EKF constructor, replacing the standard force model.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

# Define a LEO circular orbit
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
r = bh.R_EARTH + 500e3
v = (bh.GM_EARTH / r) ** 0.5
state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

# Initial covariance
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])


# Define custom two-body dynamics as additional acceleration
def two_body_dynamics(t, state, params):
    """Custom two-body gravitational acceleration."""
    pos = state[:3]
    r_mag = np.linalg.norm(pos)
    accel = -bh.GM_EARTH / r_mag**3 * pos
    deriv = np.zeros(6)
    deriv[:3] = state[3:6]  # velocity
    deriv[3:6] = accel  # acceleration
    return deriv


# Create EKF with custom dynamics via additional_dynamics parameter
ekf = bh.ExtendedKalmanFilter(
    epoch,
    state,
    p0,
    measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
    additional_dynamics=two_body_dynamics,
)

# Process a single observation using truth position
obs = bh.Observation(epoch + 60.0, state[:3], model_index=0)
record = ekf.process_observation(obs)

print("Custom dynamics EKF:")
print(f"  Prefit residual norm: {np.linalg.norm(record.prefit_residual):.3f} m")
print(f"  Postfit residual norm: {np.linalg.norm(record.postfit_residual):.6f} m")
print(f"  State dim: {len(ekf.current_state())}")
