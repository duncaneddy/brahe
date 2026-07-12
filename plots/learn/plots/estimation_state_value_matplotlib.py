"""
EKF State Value Grid - Matplotlib Backend

Plots the estimated state values with truth reference lines and 3-sigma
uncertainty patches, showing EKF convergence from a perturbed initial state.
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

# Truth orbit: LEO 500 km, ISS-like inclination
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(51.6), 0.0, 0.0, 0.0])
true_state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)

# Truth propagator
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

truth_traj = truth_prop.trajectory

# EKF with perturbed initial state (+1 km in X)
initial_state = true_state.copy()
initial_state[0] += 1000.0
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

q = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
ekf_config = bh.EKFConfig(process_noise=bh.ProcessNoiseConfig(q, scale_with_dt=True))

ekf = bh.ExtendedKalmanFilter(
    epoch,
    initial_state,
    p0,
    measurement_models=[bh.ECEFPositionMeasurementModel(5.0)],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
    config=ekf_config,
)

for obs in observations:
    ekf.process_observation(obs)

# Plot state value grid — light mode
fig = bh.plot_estimator_state_value_grid(
    solvers=[ekf],
    true_trajectory=truth_traj,
    sigma=3,
    state_labels=["X [m]", "Y [m]", "Z [m]", "Vx [m/s]", "Vy [m/s]", "Vz [m/s]"],
    labels=["EKF"],
    backend="matplotlib",
)

light_path = OUTDIR / f"{SCRIPT_NAME}_light.svg"
fig.savefig(light_path, dpi=300, bbox_inches="tight")
print(f"✓ Generated {light_path}")
plt.close(fig)

# Plot state value grid — dark mode
fig = bh.plot_estimator_state_value_grid(
    solvers=[ekf],
    true_trajectory=truth_traj,
    sigma=3,
    state_labels=["X [m]", "Y [m]", "Z [m]", "Vx [m/s]", "Vy [m/s]", "Vz [m/s]"],
    labels=["EKF"],
    backend="matplotlib",
    backend_config={"dark_mode": True},
)

dark_path = OUTDIR / f"{SCRIPT_NAME}_dark.svg"
fig.savefig(dark_path, dpi=300, bbox_inches="tight")
print(f"✓ Generated {dark_path}")
plt.close(fig)
