# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates adaptive-step integration with automatic error control.

This example shows how DP54 automatically adjusts step size to maintain
specified error tolerances.
"""

import brahe as bh
import numpy as np

# Define dynamics: Van der Pol oscillator (stiff for large mu)
mu = 1.0


def dynamics(t, state):
    x, v = state
    return np.array([v, mu * (1 - x**2) * v - x])


# Initial conditions
t0 = 0.0
state0 = np.array([2.0, 0.0])
t_end = 10.0

# Create adaptive integrator
abs_tol = 1e-8
rel_tol = 1e-7
config = bh.IntegratorConfig.adaptive(abs_tol=abs_tol, rel_tol=rel_tol)
integrator = bh.DP54Integrator(2, dynamics, config=config)

print(f"Adaptive integration of Van der Pol oscillator (μ={mu})")
print(f"Tolerances: abs_tol={abs_tol}, rel_tol={rel_tol}")
print(f"Integration time: 0 to {t_end} seconds\n")

# Integrate with adaptive stepping
t = t0
state = state0.copy()
dt = 0.1  # Initial guess
steps = 0
min_dt = float("inf")
max_dt = 0.0

print("   Time    State              Step Size   Error Est")
print("-" * 65)

while t < t_end:
    result = integrator.step(t, state, min(dt, t_end - t))

    # Track step size statistics
    min_dt = min(min_dt, result.dt_used)
    max_dt = max(max_dt, result.dt_used)

    # Update state
    t += result.dt_used
    state = result.state
    dt = result.dt_next
    steps += 1

    # Print progress
    if steps % 10 == 1:
        print(
            f"{t:7.3f}    [{state[0]:6.3f}, {state[1]:6.3f}]    {result.dt_used:7.4f}     {result.error_estimate:.2e}"
        )

print("\nIntegration complete!")
print(f"Total steps: {steps}")
print(f"Min step size: {min_dt:.6f} s")
print(f"Max step size: {max_dt:.6f} s")
print(f"Average step: {t_end / steps:.6f} s")
print("\nAdaptive stepping automatically adjusted step size")
print(f"by {max_dt / min_dt:.1f}x during integration")
