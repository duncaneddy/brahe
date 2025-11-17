#!/usr/bin/env python3
"""
Simple fixed-step integration pattern example.

Demonstrates the basic pattern for using a fixed-step integrator
with exponential decay dynamics.
"""

import brahe as bh
import numpy as np


def dynamics(t, state):
    """Exponential decay dynamics: dx/dt = -k*x"""
    k = 0.1
    return np.array([-k * state[0]])


# Create fixed-step integrator
config = bh.IntegratorConfig.fixed_step(step_size=10.0)
integrator = bh.RK4Integrator(1, dynamics, config=config)

# Integrate one step
t = 0.0
initial_state = np.array([1.0])
new_state = integrator.step(t, initial_state, dt=10.0)

print(f"Initial state: {initial_state[0]:.6f}")
print(f"State after 10s: {new_state[0]:.6f}")
print(f"Analytical: {initial_state[0] * np.exp(-0.1 * 10.0):.6f}")
