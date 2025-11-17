#!/usr/bin/env python3
"""
Multi-step propagation pattern example.

Demonstrates the pattern for propagating over an extended time period
using an adaptive integrator, using the recommended step size from
each step for the next step.
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

# --8<-- [start:snippet]
# Propagate from t=0 to t_end
t = 0.0
t_end = 86400.0  # One day
state = np.array([1.0])
dt = 60.0

step_count = 0
while t < t_end:
    result = integrator.step(t, state, min(dt, t_end - t))
    t += result.dt_used
    state = result.state
    dt = result.dt_next
    step_count += 1
# --8<-- [end:snippet]

print(f"Propagated from 0 to {t_end}s in {step_count} steps")
print(f"Final state: {state[0]:.6e}")
print(f"Analytical: {np.exp(-0.1 * t_end):.6e}")
print(f"Error: {abs(state[0] - np.exp(-0.1 * t_end)):.2e}")
