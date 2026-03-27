# /// script
# dependencies = ["brahe", "numpy", "matplotlib"]
# ///
"""
Plot BLS 2D covariance ellipse with Monte Carlo scatter using matplotlib.

Demonstrates the plot_estimator_marginal function showing the marginal
distribution of the X-Y position estimate from a Batch Least Squares
solution, with a Monte Carlo scatter overlay and 3-sigma ellipse.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

# Truth orbit: LEO 500 km, ISS-like inclination
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(51.6), 0.0, 0.0, 0.0])
true_state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)

# Truth propagator for generating simulated inertial position measurements
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
    noisy_pos = truth_eci[:3] + np.random.randn(3) * 10.0  # 10m noise
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

# Plot 2D marginal with 3-sigma covariance ellipse and MC scatter
fig = bh.plot_estimator_marginal(
    solvers=[bls],
    state_indices=(0, 1),
    sigma=3,
    state_labels=("X Position [m]", "Y Position [m]"),
    scatter_points=mc_samples,
    labels=["BLS"],
    backend="matplotlib",
)
fig.savefig("marginal_xy.svg")

pos_error = np.linalg.norm(final_state[:3] - true_state[:3])
print(f"Converged: {bls.converged()}")
print(f"Position error: {pos_error:.4f} m")
print("Saved marginal_xy.svg")
