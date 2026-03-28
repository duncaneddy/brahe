"""
EKF vs UKF State Error Comparison - Matplotlib Backend

Runs both an Extended Kalman Filter and Unscented Kalman Filter on the same
observation data, then overlays their state errors on a single grid for
direct comparison of filter convergence behaviour.
"""

import os
import pathlib
import numpy as np
import brahe as bh
import matplotlib.pyplot as plt

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Initialize EOP data
bh.initialize_eop()
bh.initialize_sw()

# Truth orbit: LEO 500 km, ISS-like inclination
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(51.6), 0.0, 0.0, 0.0])
true_state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)

# Truth propagator: full force model (20x20 gravity, drag, SRP, third-body)
# params: [mass_kg, drag_area_m2, Cd, srp_area_m2, Cr]
true_params = np.array([1000.0, 10.0, 2.2, 10.0, 1.3])
truth_prop = bh.NumericalOrbitPropagator(
    epoch,
    true_state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.default(),
    params=true_params,
)

# Generate noisy ECEF position+velocity observations every 5 minutes
# Wider spacing lets dynamics nonlinearity accumulate between updates
np.random.seed(42)
observations = []
for i in range(1, 21):
    t = epoch + i * 300.0
    truth_prop.propagate_to(t)
    truth_eci = truth_prop.current_state()
    truth_ecef = bh.state_eci_to_ecef(t, truth_eci)
    noisy_state = truth_ecef.copy()
    noisy_state[:3] += np.random.randn(3) * 50.0
    noisy_state[3:] += np.random.randn(3) * 0.5
    observations.append(bh.Observation(t, noisy_state, model_index=0))

truth_traj = truth_prop.trajectory

# Shared initial conditions: large perturbation (+50 km in X, +10 m/s in Vy)
# to stress-test linearization and reveal EKF vs UKF differences
initial_state = true_state.copy()
initial_state[0] += 50e3
initial_state[4] += 10.0
p0 = np.diag([1e8, 1e8, 1e8, 1e4, 1e4, 1e4])

q = np.diag([1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4])

# Both filters use 5x5 gravity only (no drag/SRP) — model mismatch with truth
filter_force = bh.ForceModelConfig(gravity=bh.GravityConfiguration(degree=5, order=5))

# EKF
ekf_config = bh.EKFConfig(process_noise=bh.ProcessNoiseConfig(q, scale_with_dt=True))
ekf = bh.ExtendedKalmanFilter(
    epoch,
    initial_state,
    p0,
    measurement_models=[bh.ECEFStateMeasurementModel(50.0, 0.5)],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=filter_force,
    config=ekf_config,
)

for obs in observations:
    ekf.process_observation(obs)

# UKF (same initial conditions and process noise)
ukf_config = bh.UKFConfig(process_noise=bh.ProcessNoiseConfig(q, scale_with_dt=True))
ukf = bh.UnscentedKalmanFilter(
    epoch,
    initial_state,
    p0,
    measurement_models=[bh.ECEFStateMeasurementModel(50.0, 0.5)],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=filter_force,
    config=ukf_config,
)

for obs in observations:
    ukf.process_observation(obs)

# Plot both filters on the same grid — light mode
fig = bh.plot_estimator_state_error_grid(
    solvers=[ekf, ukf],
    true_trajectory=truth_traj,
    sigma=3,
    state_labels=["X [m]", "Y [m]", "Z [m]", "Vx [m/s]", "Vy [m/s]", "Vz [m/s]"],
    labels=["EKF", "UKF"],
    colors=["#1f77b4", "#d62728"],
    backend="matplotlib",
)

light_path = OUTDIR / f"{SCRIPT_NAME}_light.svg"
fig.savefig(light_path, dpi=300, bbox_inches="tight")
print(f"✓ Generated {light_path}")
plt.close(fig)

# Plot both filters on the same grid — dark mode
fig = bh.plot_estimator_state_error_grid(
    solvers=[ekf, ukf],
    true_trajectory=truth_traj,
    sigma=3,
    state_labels=["X [m]", "Y [m]", "Z [m]", "Vx [m/s]", "Vy [m/s]", "Vz [m/s]"],
    labels=["EKF", "UKF"],
    colors=["#1f77b4", "#d62728"],
    backend="matplotlib",
    backend_config={"dark_mode": True},
)

dark_path = OUTDIR / f"{SCRIPT_NAME}_dark.svg"
fig.savefig(dark_path, dpi=300, bbox_inches="tight")
print(f"✓ Generated {dark_path}")
plt.close(fig)
