# /// script
# dependencies = ["brahe", "numpy", "plotly"]
# ///
"""
Plot a grid of EKF state values versus truth trajectory using plotly.

Demonstrates the plot_estimator_state_value_grid function with truth reference
lines and 3-sigma covariance bands, showing estimated position and velocity
components converging to truth from a perturbed initial state.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

# Truth orbit: LEO 500 km, ISS-like inclination
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(51.6), 0.0, 0.0, 0.0])
true_state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)

# Truth propagator — also used to build the truth trajectory
truth_prop = bh.NumericalOrbitPropagator(
    epoch,
    true_state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
)

# Generate noisy ECEF GNSS position observations
np.random.seed(42)
observations = []
for i in range(1, 31):
    t = epoch + i * 60.0
    truth_prop.propagate_to(t)
    truth_eci = truth_prop.current_state()
    truth_ecef_pos = bh.position_eci_to_ecef(t, truth_eci[:3])
    noisy_pos = truth_ecef_pos + np.random.randn(3) * 5.0
    observations.append(bh.Observation(t, noisy_pos, model_index=0))

# Truth trajectory for reference lines in the value plot
truth_traj = truth_prop.trajectory

# EKF with perturbed initial state (+1 km in X)
initial_state = true_state.copy()
initial_state[0] += 1000.0
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

ekf = bh.ExtendedKalmanFilter(
    epoch,
    initial_state,
    p0,
    measurement_models=[bh.ECEFPositionMeasurementModel(5.0)],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
)

for obs in observations:
    ekf.process_observation(obs)

# Plot 6-panel state value grid with truth reference and 3-sigma bands
fig = bh.plot_estimator_state_value_grid(
    solvers=[ekf],
    true_trajectory=truth_traj,
    sigma=3,
    state_labels=["X [m]", "Y [m]", "Z [m]", "Vx [m/s]", "Vy [m/s]", "Vz [m/s]"],
    labels=["EKF"],
    backend="plotly",
)
fig.write_html("state_value_grid.html")

final_state = ekf.current_state()
pos_error = np.linalg.norm(final_state[:3] - truth_prop.current_state()[:3])
print(f"Observations processed: {len(ekf.records())}")
print(f"Final position error: {pos_error:.2f} m")
print("Saved state_value_grid.html")
