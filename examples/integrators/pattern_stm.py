#!/usr/bin/env python3
"""
State Transition Matrix propagation pattern example.

Demonstrates the basic pattern for propagating a state transition matrix
alongside the state using variational equations.
"""

import brahe as bh
import numpy as np


def dynamics(t, state):
    """Exponential decay dynamics: dx/dt = -k*x"""
    k = 0.1
    return np.array([-k * state[0]])


# Create Jacobian for variational equations
jacobian = bh.NumericalJacobian.central(dynamics).with_adaptive(
    scale_factor=1e-8, min_value=1e-6
)

# Create integrator with Jacobian
config = bh.IntegratorConfig.adaptive(abs_tol=1e-12, rel_tol=1e-11)
integrator = bh.DP54Integrator(1, dynamics, jacobian=jacobian, config=config)

# Propagate state and STM
t = 0.0
state = np.array([1.0])
phi = np.eye(1)  # Identity matrix
dt = 60.0

new_state, new_phi, dt_used, error_est, dt_next = integrator.step_with_varmat(
    t, state, phi, dt
)

print(f"Initial state: {state[0]:.6f}")
print(f"State after {dt_used:.2f}s: {new_state[0]:.6f}")
print("State transition matrix:")
print(f"  Φ = {new_phi[0, 0]:.6f}")
print(f"  (Analytical Φ = {np.exp(-0.1 * dt_used):.6f})")
