# /// script
# dependencies = ["brahe", "numpy", "plotly"]
# ///
"""
Plot BLS measurement residual RMS over time using plotly.

Demonstrates the plot_measurement_residual_rms function showing the root
mean square of postfit residuals from a Batch Least Squares solution,
providing a scalar summary of fit quality across the observation window.
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

# Plot postfit residual RMS as a single scalar time series
fig = bh.plot_measurement_residual_rms(
    solver=bls,
    residual_type="postfit",
    backend="plotly",
)
fig.write_html("residual_rms.html")

print(f"Converged: {bls.converged()}")
print(f"Iterations: {bls.iterations_completed()}")
print(f"Final cost: {bls.final_cost():.6e}")
print("Saved residual_rms.html")
