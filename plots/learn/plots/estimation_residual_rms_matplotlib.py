"""
EKF Residual RMS - Matplotlib Backend

Plots the root mean square of postfit residuals from an Extended Kalman Filter,
providing a scalar summary of fit quality over the observation window.
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

# Initialize EOP and space weather data
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

# Generate noisy inertial position observations
np.random.seed(42)
observations = []
for i in range(1, 31):
    t = epoch + i * 60.0
    truth_prop.propagate_to(t)
    truth_eci = truth_prop.current_state()
    noisy_pos = truth_eci[:3] + np.random.randn(3) * 10.0
    observations.append(bh.Observation(t, noisy_pos, model_index=0))

# EKF with perturbed initial state (+1 km in X) and simplified dynamics
initial_state = true_state.copy()
initial_state[0] += 1000.0
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

q = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
ekf_config = bh.EKFConfig(process_noise=bh.ProcessNoiseConfig(q, scale_with_dt=True))

ekf = bh.ExtendedKalmanFilter(
    epoch,
    initial_state,
    p0,
    measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig(
        gravity=bh.GravityConfiguration(degree=5, order=5)
    ),
    config=ekf_config,
)

for obs in observations:
    ekf.process_observation(obs)

# Plot residual RMS — light mode
fig = bh.plot_measurement_residual_rms(
    solver=ekf,
    residual_type="postfit",
    backend="matplotlib",
)

light_path = OUTDIR / f"{SCRIPT_NAME}_light.svg"
fig.savefig(light_path, dpi=300, bbox_inches="tight")
print(f"✓ Generated {light_path}")
plt.close(fig)

# Plot residual RMS — dark mode
fig = bh.plot_measurement_residual_rms(
    solver=ekf,
    residual_type="postfit",
    backend="matplotlib",
    backend_config={"dark_mode": True},
)

dark_path = OUTDIR / f"{SCRIPT_NAME}_dark.svg"
fig.savefig(dark_path, dpi=300, bbox_inches="tight")
print(f"✓ Generated {dark_path}")
plt.close(fig)
