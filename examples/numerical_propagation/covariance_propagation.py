# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Covariance propagation using the State Transition Matrix.
Demonstrates propagating initial uncertainty through orbital dynamics.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Create propagation config with STM enabled
prop_config = bh.NumericalPropagationConfig.default().with_stm()

# Create propagator (two-body for clean demonstration)
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    prop_config,
    bh.ForceModelConfig.two_body(),
    None,
)

# Define initial covariance (diagonal)
# Position uncertainty: 10 m in each axis
# Velocity uncertainty: 0.01 m/s in each axis
P0 = np.diag([100.0, 100.0, 100.0, 0.0001, 0.0001, 0.0001])

print("Initial Covariance (diagonal, sqrt):")
print(f"  Position std: {np.sqrt(P0[0, 0]):.1f} m")
print(f"  Velocity std: {np.sqrt(P0[3, 3]) * 1000:.2f} mm/s")

# Propagate for one orbital period
orbital_period = bh.orbital_period(oe[0])
prop.propagate_to(epoch + orbital_period)

# Get the State Transition Matrix
stm = prop.stm()
print(f"\nSTM shape: {stm.shape}")

# Propagate covariance: P(t) = Phi @ P0 @ Phi^T
P = stm @ P0 @ stm.T

# Extract position and velocity uncertainties
pos_cov = P[:3, :3]
vel_cov = P[3:, 3:]

print("\nPropagated Covariance after one orbit:")
print(
    f"  Position std (x,y,z): ({np.sqrt(pos_cov[0, 0]):.1f}, {np.sqrt(pos_cov[1, 1]):.1f}, {np.sqrt(pos_cov[2, 2]):.1f}) m"
)
print(
    f"  Velocity std (x,y,z): ({np.sqrt(vel_cov[0, 0]) * 1000:.2f}, {np.sqrt(vel_cov[1, 1]) * 1000:.2f}, {np.sqrt(vel_cov[2, 2]) * 1000:.2f}) mm/s"
)

# Compute position uncertainty magnitude
pos_uncertainty_initial = np.sqrt(np.trace(P0[:3, :3]))
pos_uncertainty_final = np.sqrt(np.trace(pos_cov))

print("\nTotal position uncertainty:")
print(f"  Initial: {pos_uncertainty_initial:.1f} m")
print(f"  Final:   {pos_uncertainty_final:.1f} m")
print(f"  Growth:  {pos_uncertainty_final / pos_uncertainty_initial:.1f}x")

# Validate that covariance was propagated
assert stm is not None
assert stm.shape == (6, 6)
assert pos_uncertainty_final >= pos_uncertainty_initial  # Uncertainty grows

print("\nExample validated successfully!")
