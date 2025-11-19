#!/usr/bin/env python3
"""
Simple adaptive-step integration pattern example.

Demonstrates the basic pattern for using an adaptive-step integrator
with exponential decay dynamics.
"""

import brahe as bh
import numpy as np


def dynamics(t, state):
    """Exponential decay dynamics: dx/dt = -k*x"""
    k = 0.1
    return np.array([-k * state[0]])


# Create adaptive integrator
config = bh.IntegratorConfig.adaptive(abs_tol=1e-10, rel_tol=1e-9)
integrator = bh.DP54Integrator(1, dynamics, config=config)

# Integrate with automatic step control
t = 0.0
initial_state = np.array([1.0])
dt = 60.0  # Initial guess

result = integrator.step(t, initial_state, dt)

print(f"Initial state: {initial_state[0]:.6f}")
print(f"State after step: {result.state[0]:.6f}")
print(f"Step used: {result.dt_used:.2f}s")
print(f"Recommended next step: {result.dt_next:.2f}s")
print(f"Error estimate: {result.error_estimate:.2e}")
