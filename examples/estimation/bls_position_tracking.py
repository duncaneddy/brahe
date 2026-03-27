# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Batch Least Squares orbit determination from position measurements.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

# Define truth orbit: LEO circular at ~500km altitude
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
r = bh.R_EARTH + 500e3
v = (bh.GM_EARTH / r) ** 0.5
true_state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

# Generate truth trajectory and noise-free position observations
truth_prop = bh.NumericalOrbitPropagator(
    epoch,
    true_state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
)

observations = []
for i in range(1, 21):
    t = epoch + i * 30.0
    truth_prop.propagate_to(t)
    pos = truth_prop.current_state()[:3]
    observations.append(bh.Observation(t, pos, 0))

# Initial guess: perturbed by 1km in x-position
initial_state = true_state.copy()
initial_state[0] += 1000.0

# A priori covariance
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

# Create and run BLS
bls = bh.BatchLeastSquares(
    epoch,
    initial_state,
    p0,
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
    measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
)

bls.solve(observations)

# Results
print(f"Converged: {bls.converged()}")
print(f"Iterations: {bls.iterations_completed()}")
print(f"Final cost: {bls.final_cost():.6e}")

pos_error = np.linalg.norm(bls.current_state()[:3] - true_state[:3])
print(f"Position error: {pos_error:.6f} m")

# Iteration history
for rec in bls.iteration_records():
    print(
        f"  Iter {rec.iteration}: cost={rec.cost:.6e}, ||dx||={rec.state_correction_norm:.6e}"
    )
