# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates using analytical Jacobian computation for a simple harmonic oscillator.

This example shows how to provide closed-form derivatives when they're known.
"""

import brahe as bh
import numpy as np


# Define dynamics: Simple harmonic oscillator
# State: [position, velocity]
# Dynamics: dx/dt = v, dv/dt = -x
def dynamics(t, state):
    return np.array([state[1], -state[0]])


# Define analytical Jacobian
# J = [[0,  1],
#      [-1, 0]]
def jacobian_func(t, state):
    return np.array([[0.0, 1.0], [-1.0, 0.0]])


# Create analytical Jacobian provider
jacobian = bh.AnalyticJacobian(jacobian_func)

# Compute Jacobian at a specific state
t = 0.0
state = np.array([1.0, 0.0])
J = jacobian.compute(t, state)

print("Analytical Jacobian:")
print(J)
# Expected output:
# [[ 0.  1.]
#  [-1.  0.]]

# Verify it's time-invariant for this system
t2 = 10.0
state2 = np.array([0.5, 0.866])
J2 = jacobian.compute(t2, state2)

print("\nJacobian at different time and state:")
print(J2)
# Should be identical for linear system

print("\nJacobians are equal:", np.allclose(J, J2))
# Output: True
