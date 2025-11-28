# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Overview of variational propagation: STM, covariance, and sensitivity.
Demonstrates enabling and using all variational features together.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state (LEO satellite)
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Define spacecraft parameters: [mass, drag_area, Cd, srp_area, Cr]
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

# Create propagation config enabling STM and sensitivity with history storage
prop_config = (
    bh.NumericalPropagationConfig.default()
    .with_stm()
    .with_stm_history()
    .with_sensitivity()
    .with_sensitivity_history()
)

# Define initial covariance (diagonal)
# Position uncertainty: 10 m (variance = 100 m²)
# Velocity uncertainty: 0.01 m/s (variance = 0.0001 m²/s²)
P0 = np.diag([100.0, 100.0, 100.0, 0.0001, 0.0001, 0.0001])

# Create propagator with full force model
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    prop_config,
    bh.ForceModelConfig.default(),
    params=params,
    initial_covariance=P0,
)

print("=== Variational Propagation Overview ===\n")
print("Initial State:")
print(f"  Semi-major axis: {oe[0] / 1000:.1f} km")
print(f"  Position std: {np.sqrt(P0[0, 0]):.1f} m")
print(f"  Velocity std: {np.sqrt(P0[3, 3]) * 1000:.2f} mm/s")

# Propagate for one orbital period
orbital_period = bh.orbital_period(oe[0])
prop.propagate_to(epoch + orbital_period)

# === STM Access ===
print("\n--- State Transition Matrix (STM) ---")
stm = prop.stm()
print(f"STM shape: {stm.shape}")
print(
    f"STM determinant: {np.linalg.det(stm):.6f} (should be ~1 for conservative forces)"
)

# STM at intermediate time (half orbit)
stm_half = prop.stm_at(epoch + orbital_period / 2)
print(f"STM at t/2 available: {stm_half is not None}")

# === Covariance Propagation ===
print("\n--- Covariance Propagation ---")

# Manual propagation: P(t) = STM @ P0 @ STM^T
P_manual = stm @ P0 @ stm.T

# Using built-in covariance retrieval
P_gcrf = prop.covariance_gcrf(epoch + orbital_period)
P_rtn = prop.covariance_rtn(epoch + orbital_period)

# Extract position uncertainties
pos_std_gcrf = np.sqrt(np.diag(P_gcrf[:3, :3]))
pos_std_rtn = np.sqrt(np.diag(P_rtn[:3, :3]))

print("Position std (GCRF frame):")
print(
    f"  X: {pos_std_gcrf[0]:.1f} m, Y: {pos_std_gcrf[1]:.1f} m, Z: {pos_std_gcrf[2]:.1f} m"
)
print("Position std (RTN frame):")
print(
    f"  R: {pos_std_rtn[0]:.1f} m, T: {pos_std_rtn[1]:.1f} m, N: {pos_std_rtn[2]:.1f} m"
)

# === Sensitivity Analysis ===
print("\n--- Parameter Sensitivity ---")
sens = prop.sensitivity()
print(f"Sensitivity matrix shape: {sens.shape}")

# Position sensitivity magnitude to each parameter
param_names = ["mass", "drag_area", "Cd", "srp_area", "Cr"]
print("\nPosition sensitivity to 1% parameter uncertainty:")
for i, name in enumerate(param_names):
    pos_sens_mag = np.linalg.norm(sens[:3, i])
    param_uncertainty = params[i] * 0.01  # 1% uncertainty
    pos_error = pos_sens_mag * param_uncertainty
    print(f"  {name:10s}: {pos_error:.2f} m")

# === Summary ===
print("\n--- Summary ---")
total_pos_std_initial = np.sqrt(np.trace(P0[:3, :3]))
total_pos_std_final = np.sqrt(np.trace(P_gcrf[:3, :3]))
print(
    f"Total position uncertainty: {total_pos_std_initial:.1f} m -> {total_pos_std_final:.1f} m"
)
print(f"Uncertainty growth factor: {total_pos_std_final / total_pos_std_initial:.1f}x")

# Validate outputs
assert stm.shape == (6, 6)
assert sens.shape == (6, 5)
assert P_gcrf.shape == (6, 6)
assert P_rtn.shape == (6, 6)
assert total_pos_std_final >= total_pos_std_initial

print("\nExample validated successfully!")
