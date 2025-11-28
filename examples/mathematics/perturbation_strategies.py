# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compares different perturbation strategies for numerical Jacobian computation.

This example demonstrates how to choose appropriate step sizes for states
with different magnitudes.
"""

import brahe as bh
import numpy as np


# Define dynamics with mixed-scale state
# State: [large_position (km), small_velocity (km/s)]
def dynamics(t, state):
    # Simple dynamics with different scales
    x, v = state
    return np.array([v, -x * 1e-6])  # Different scales


# Analytical Jacobian
def analytical_jacobian(t, state):
    return np.array([[0.0, 1.0], [-1e-6, 0.0]])


# Test state with very different magnitudes
state = np.array([7000.0, 7.5])  # Position in km, velocity in km/s
t = 0.0

J_analytical = analytical_jacobian(t, state)

print("Testing perturbation strategies on mixed-scale state:")
print(f"State: position={state[0]} km, velocity={state[1]} km/s\n")

# Strategy 1: Fixed perturbation
print("1. Fixed Perturbation (h = 1e-6)")
jacobian_fixed = bh.NumericalJacobian.central(dynamics).with_fixed_offset(1e-6)
J_fixed = jacobian_fixed.compute(t, state)
error_fixed = np.linalg.norm(J_fixed - J_analytical)
print(f"   Error: {error_fixed:.2e}\n")

# Strategy 2: Percentage perturbation
print("2. Percentage Perturbation (0.001%)")
jacobian_pct = bh.NumericalJacobian.central(dynamics).with_percentage(1e-5)
J_pct = jacobian_pct.compute(t, state)
error_pct = np.linalg.norm(J_pct - J_analytical)
print(f"   Error: {error_pct:.2e}\n")

# Strategy 3: Adaptive perturbation (recommended)
print("3. Adaptive Perturbation (scale=1.0, min=1.0)")
jacobian_adaptive = bh.NumericalJacobian.central(dynamics).with_adaptive(
    scale_factor=1.0, min_value=1.0
)
J_adaptive = jacobian_adaptive.compute(t, state)
error_adaptive = np.linalg.norm(J_adaptive - J_analytical)
print(f"   Error: {error_adaptive:.2e}\n")

# Summary
print("Strategy Comparison:")
print(f"  Fixed:      {error_fixed:.2e}")
print(f"  Percentage: {error_pct:.2e}")
print(f"  Adaptive:   {error_adaptive:.2e}")
print("\nBest strategy: Adaptive (handles mixed scales robustly)")

# Test with state component near zero
print("\n" + "=" * 60)
print("Testing with component near zero:")
state_zero = np.array([7000.0, 1e-9])  # Very small velocity
print(f"State: position={state_zero[0]} km, velocity={state_zero[1]} km/s\n")

J_analytical_zero = analytical_jacobian(t, state_zero)

# Percentage fails when component is near zero
try:
    J_pct_zero = jacobian_pct.compute(t, state_zero)
    error_pct_zero = np.linalg.norm(J_pct_zero - J_analytical_zero)
    print(f"Percentage: Error = {error_pct_zero:.2e}")
except ZeroDivisionError:
    print("Percentage: FAILED (division by near-zero)")

# Adaptive handles it gracefully
J_adaptive_zero = jacobian_adaptive.compute(t, state_zero)
error_adaptive_zero = np.linalg.norm(J_adaptive_zero - J_analytical_zero)
print(f"Adaptive:   Error = {error_adaptive_zero:.2e}")

print("\nConclusion: Adaptive perturbation is most robust")
