# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
State Transition Matrix (STM) propagation.
Demonstrates enabling STM computation and accessing results.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Method 1: Enable STM via builder pattern
prop_config = bh.NumericalPropagationConfig.default().with_stm().with_stm_history()

# Create propagator with two-body gravity
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    prop_config,
    bh.ForceModelConfig.two_body(),
    None,
)

print("=== STM Propagation Example ===\n")

# Propagate for one orbital period
orbital_period = bh.orbital_period(oe[0])
prop.propagate_to(epoch + orbital_period)

# Access STM at final time
stm = prop.stm()
print(f"Final STM shape: {stm.shape}")
print(f"STM determinant: {np.linalg.det(stm):.6f}")

# STM at initial time should be identity
stm_initial = prop.stm_at(epoch)
print("\nSTM at t=0 (should be identity):")
print(f"  Max off-diagonal: {np.max(np.abs(stm_initial - np.eye(6))):.2e}")

# STM at intermediate time
half_period = epoch + orbital_period / 2
stm_half = prop.stm_at(half_period)
print("\nSTM at t=T/2:")
print(f"  Determinant: {np.linalg.det(stm_half):.6f}")

# STM composition property: Phi(t2,t0) = Phi(t2,t1) * Phi(t1,t0)
# For verification, we check that the STM is invertible
stm_inv = np.linalg.inv(stm)
identity_check = stm @ stm_inv
print("\nSTM * STM^-1 (should be identity):")
print(f"  Max deviation from I: {np.max(np.abs(identity_check - np.eye(6))):.2e}")

# STM structure interpretation
print("\n=== STM Structure ===")
print("Upper-left 3x3: Position sensitivity to initial position")
print("Upper-right 3x3: Position sensitivity to initial velocity")
print("Lower-left 3x3: Velocity sensitivity to initial position")
print("Lower-right 3x3: Velocity sensitivity to initial velocity")

# Show magnitude of each block
pos_pos = np.linalg.norm(stm[:3, :3])
pos_vel = np.linalg.norm(stm[:3, 3:])
vel_pos = np.linalg.norm(stm[3:, :3])
vel_vel = np.linalg.norm(stm[3:, 3:])

print("\nBlock Frobenius norms after one orbit:")
print(f"  dr/dr0: {pos_pos:.2f}")
print(f"  dr/dv0: {pos_vel:.2f}")
print(f"  dv/dr0: {vel_pos:.6f}")
print(f"  dv/dv0: {vel_vel:.2f}")

# Validate
assert stm.shape == (6, 6)
assert np.abs(np.linalg.det(stm) - 1.0) < 1e-6  # Hamiltonian system preserves volume
assert stm_initial is not None
assert stm_half is not None

print("\nExample validated successfully!")
