# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates numerical Jacobian computation using finite differences.

This example shows how to automatically compute Jacobians without deriving
analytical expressions.
"""

import brahe as bh
import numpy as np


# Define dynamics: Simple harmonic oscillator
def dynamics(t, state):
    return np.array([state[1], -state[0]])


# Create numerical Jacobian with default settings (central differences)
jacobian = bh.NumericalJacobian(dynamics)

# Compute Jacobian at a specific state
t = 0.0
state = np.array([1.0, 0.0])
J_numerical = jacobian.compute(t, state)

print("Numerical Jacobian (central differences):")
print(J_numerical)
# Expected output (should be very close to analytical):
# [[ 0.  1.]
#  [-1.  0.]]

# Compare with analytical solution
J_analytical = np.array([[0.0, 1.0], [-1.0, 0.0]])

error = np.linalg.norm(J_numerical - J_analytical)
print(f"\nError vs analytical: {error:.2e}")
# Output: ~1e-8 (machine precision)

# Verify accuracy at different state
state2 = np.array([0.5, 0.866])
J_numerical2 = jacobian.compute(t, state2)

print("\nNumerical Jacobian at different state:")
print(J_numerical2)

error2 = np.linalg.norm(J_numerical2 - J_analytical)
print(f"Error vs analytical: {error2:.2e}")
# Output: ~1e-8 (consistent accuracy)
