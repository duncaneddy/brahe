# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compares different finite difference methods for numerical Jacobian computation.

This example demonstrates the accuracy trade-offs between forward, central,
and backward difference methods using two-body gravity dynamics.
"""

import brahe as bh
import numpy as np


# Define two-body gravity dynamics: state = [x, y, z, vx, vy, vz]
def gravity_dynamics(t, state):
    r = state[0:3]  # Position
    v = state[3:6]  # Velocity
    r_norm = np.linalg.norm(r)

    # Acceleration from two-body gravity: a = -mu * r / |r|^3
    a = -bh.GM_EARTH * r / r_norm**3

    return np.concatenate([v, a])


# Analytical Jacobian for two-body gravity
def analytical_jacobian(state):
    r = state[0:3]
    r_norm = np.linalg.norm(r)
    r3 = r_norm**3
    r5 = r_norm**5

    # Top-left: zeros (3x3)
    # Top-right: identity (3x3)
    # Bottom-left: gravity gradient (3x3)
    # Bottom-right: zeros (3x3)
    J = np.zeros((6, 6))
    J[0:3, 3:6] = np.eye(3)  # Velocity contribution to position derivative

    # Gravity gradient term - Motenbruck Eqn 7.56
    J[3:6, 0:3] = -bh.GM_EARTH * (np.eye(3) / r3 - 3 * np.outer(r, r) / r5)

    return J


# Create numerical Jacobians with different methods
jacobian_forward = bh.NumericalJacobian.forward(gravity_dynamics)
jacobian_central = bh.NumericalJacobian.central(gravity_dynamics)
jacobian_backward = bh.NumericalJacobian.backward(gravity_dynamics)

# Test state: Low Earth Orbit position and velocity
t = 0.0
state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])  # Circular orbit

# Compute analytical Jacobian
J_analytical = analytical_jacobian(state)

# Compute Jacobians with each method
J_forward = jacobian_forward.compute(t, state)
J_central = jacobian_central.compute(t, state)
J_backward = jacobian_backward.compute(t, state)

print("Forward Difference Jacobian:")
for row in J_forward:
    print("[" + "  ".join(f"{val: .2e}" for val in row) + "]")
error_forward = np.linalg.norm(J_forward - J_analytical)
print(f"Error: {error_forward:.2e}\n")

print("Central Difference Jacobian:")
for row in J_central:
    print("[" + "  ".join(f"{val: .2e}" for val in row) + "]")
error_central = np.linalg.norm(J_central - J_analytical)
print(f"Error: {error_central:.2e}\n")

print("Backward Difference Jacobian:")
for row in J_backward:
    print("[" + "  ".join(f"{val: .2e}" for val in row) + "]")
error_backward = np.linalg.norm(J_backward - J_analytical)
print(f"Error: {error_backward:.2e}\n")

# Summary
print("Accuracy Comparison:")
print(f"  Forward:  {error_forward:.2e} (O(h))")
print(f"  Central:  {error_central:.2e} (O(h²))")
print(f"  Backward: {error_backward:.2e} (O(h))")
print(f"\nCentral is {error_forward / error_central:.1f}x more accurate than forward")
print(f"Central is {error_backward / error_central:.1f}x more accurate than backward")

# Expected output:
# Forward Difference Jacobian:
# [ 0.00e+00   0.00e+00   0.00e+00   1.00e+00   0.00e+00   0.00e+00]
# [ 0.00e+00   0.00e+00   0.00e+00   0.00e+00   1.00e+00   0.00e+00]
# [ 0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00   1.00e+00]
# [ 2.45e-06   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00]
# [ 0.00e+00  -1.22e-06   0.00e+00   0.00e+00   0.00e+00   0.00e+00]
# [ 0.00e+00   0.00e+00  -1.22e-06   0.00e+00   0.00e+00   0.00e+00]
# Error: 5.54e-14

# Central Difference Jacobian:
# [ 0.00e+00   0.00e+00   0.00e+00   1.00e+00   0.00e+00   0.00e+00]
# [ 0.00e+00   0.00e+00   0.00e+00   0.00e+00   1.00e+00   0.00e+00]
# [ 0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00   1.00e+00]
# [ 2.45e-06   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00]
# [ 0.00e+00  -1.22e-06   0.00e+00   0.00e+00   0.00e+00   0.00e+00]
# [ 0.00e+00   0.00e+00  -1.22e-06   0.00e+00   0.00e+00   0.00e+00]
# Error: 5.27e-15

# Backward Difference Jacobian:
# [ 0.00e+00   0.00e+00   0.00e+00   1.00e+00   0.00e+00   0.00e+00]
# [ 0.00e+00   0.00e+00   0.00e+00   0.00e+00   1.00e+00   0.00e+00]
# [ 0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00   1.00e+00]
# [ 2.45e-06   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.00e+00]
# [ 0.00e+00  -1.22e-06   0.00e+00   0.00e+00   0.00e+00   0.00e+00]
# [ 0.00e+00   0.00e+00  -1.22e-06   0.00e+00   0.00e+00   0.00e+00]
# Error: 6.59e-14

# Accuracy Comparison:
#   Forward:  5.54e-14 (O(h))
#   Central:  5.27e-15 (O(h²))
#   Backward: 6.59e-14 (O(h))

# Central is 10.5x more accurate than forward
# Central is 12.5x more accurate than backward
