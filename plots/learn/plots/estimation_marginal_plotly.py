"""
BLS Marginal Distribution - Plotly Backend

Plots the 2D covariance ellipse for the X-Y position estimate with Monte Carlo
scatter overlay, showing the marginal distribution from a Batch Least Squares solution.
"""

import os
import pathlib
import sys
import numpy as np
import brahe as bh

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from brahe_theme import save_themed_html

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

# Generate noisy ECI position observations
np.random.seed(42)
observations = []
for i in range(1, 31):
    t = epoch + i * 60.0
    truth_prop.propagate_to(t)
    truth_eci = truth_prop.current_state()
    noisy_pos = truth_eci[:3] + np.random.randn(3) * 10.0
    observations.append(bh.Observation(t, noisy_pos, model_index=0))

# BLS with perturbed initial state (+1 km in X)
initial_state = true_state.copy()
initial_state[0] += 1000.0
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

bls = bh.BatchLeastSquares(
    epoch,
    initial_state,
    p0,
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
    measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
)
bls.solve(observations)

# Generate Monte Carlo samples from the final X-Y position covariance
final_state = bls.current_state()
final_cov = bls.current_covariance()
mc_samples = np.random.multivariate_normal(final_state[:2], final_cov[:2, :2], 200)

# Plot marginal
fig = bh.plot_estimator_marginal(
    solvers=[bls],
    state_indices=(0, 1),
    sigma=3,
    state_labels=("X Position [m]", "Y Position [m]"),
    scatter_points=mc_samples,
    labels=["BLS"],
    backend="plotly",
)

# Save themed HTML files
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
