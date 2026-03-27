# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Track a satellite with an Unscented Kalman Filter using position measurements.

Same scenario as the EKF example, but using the UKF which propagates sigma
points through the nonlinear dynamics instead of linearizing with STM.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

# Define a LEO circular orbit
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
r = bh.R_EARTH + 500e3
v = (bh.GM_EARTH / r) ** 0.5
true_state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

# Create a truth propagator for generating observations
truth_prop = bh.NumericalOrbitPropagator(
    epoch,
    true_state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
)

# Perturbed initial state: 1 km position error, 1 m/s velocity error
initial_state = true_state.copy()
initial_state[0] += 1000.0
initial_state[4] += 1.0

# Initial covariance reflecting our uncertainty
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

# Create the UKF with inertial position measurements (10 m noise)
ukf = bh.UnscentedKalmanFilter(
    epoch,
    initial_state,
    p0,
    measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
)

# Process 30 observations at 60-second intervals
dt = 60.0
for i in range(1, 31):
    obs_epoch = epoch + dt * i
    truth_prop.propagate_to(obs_epoch)
    truth_pos = truth_prop.current_state()[:3]
    obs = bh.Observation(obs_epoch, truth_pos, model_index=0)
    ukf.process_observation(obs)

# Compare final state to truth
truth_prop.propagate_to(ukf.current_epoch())
truth_final = truth_prop.current_state()
final_state = ukf.current_state()
pos_error = np.linalg.norm(final_state[:3] - truth_final[:3])
vel_error = np.linalg.norm(final_state[3:6] - truth_final[3:6])

print("Initial position error: 1000.0 m")
print(f"Final position error:   {pos_error:.2f} m")
print(f"Final velocity error:   {vel_error:.4f} m/s")
print(f"Observations processed: {len(ukf.records())}")

# Show final covariance diagonal (1-sigma uncertainties)
cov = ukf.current_covariance()
sigma = np.sqrt(np.diag(cov))
print("\n1-sigma uncertainties:")
print(f"  Position: [{sigma[0]:.1f}, {sigma[1]:.1f}, {sigma[2]:.1f}] m")
print(f"  Velocity: [{sigma[3]:.4f}, {sigma[4]:.4f}, {sigma[5]:.4f}] m/s")
