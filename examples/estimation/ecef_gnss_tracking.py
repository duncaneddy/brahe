# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Track a satellite using ECEF position measurements from a GNSS receiver.

Demonstrates how ECEFPositionMeasurementModel handles the ECI-to-ECEF
frame rotation internally, so GNSS receiver outputs can be used directly.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

# Define a LEO circular orbit
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
r = bh.R_EARTH + 500e3
v = (bh.GM_EARTH / r) ** 0.5
true_state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

# Truth propagator for generating simulated GNSS observations
truth_prop = bh.NumericalOrbitPropagator(
    epoch,
    true_state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
)

# Perturbed initial state: 1 km position error
initial_state = true_state.copy()
initial_state[0] += 1000.0
initial_state[4] += 1.0
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

# ECEF position model with typical GNSS accuracy (5 m noise)
ecef_model = bh.ECEFPositionMeasurementModel(5.0)

ekf = bh.ExtendedKalmanFilter(
    epoch,
    initial_state,
    p0,
    measurement_models=[ecef_model],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
)

# Simulate GNSS observations: get truth ECI state, convert to ECEF
dt = 60.0
for i in range(1, 21):
    obs_epoch = epoch + dt * i
    truth_prop.propagate_to(obs_epoch)
    truth_eci = truth_prop.current_state()

    # Simulate GNSS: convert truth position to ECEF
    truth_ecef_pos = bh.position_eci_to_ecef(obs_epoch, truth_eci[:3])

    obs = bh.Observation(obs_epoch, truth_ecef_pos, model_index=0)
    ekf.process_observation(obs)

# Compare final state to truth
truth_prop.propagate_to(ekf.current_epoch())
truth_final = truth_prop.current_state()
final_state = ekf.current_state()
pos_error = np.linalg.norm(final_state[:3] - truth_final[:3])
vel_error = np.linalg.norm(final_state[3:6] - truth_final[3:6])

print("ECEF GNSS tracking with ECEFPositionMeasurementModel:")
print("  Initial position error: 1000.0 m")
print(f"  Final position error:   {pos_error:.2f} m")
print(f"  Final velocity error:   {vel_error:.4f} m/s")
print(f"  Observations processed: {len(ekf.records())}")
