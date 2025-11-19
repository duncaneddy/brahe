# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates RK4 fixed-step integration.

This example shows the characteristics of fixed-step integration including
predictable computational cost and the importance of choosing the right step size.
"""

import brahe as bh
import numpy as np

# Define simple harmonic oscillator
omega = 1.0


def dynamics(t, state):
    x, v = state
    return np.array([v, -(omega**2) * x])


# Analytical solution
def analytical(t, x0=1.0, v0=0.0):
    x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    v = -omega * x0 * np.sin(omega * t) + v0 * np.cos(omega * t)
    return np.array([x, v])


# Initial conditions
state0 = np.array([1.0, 0.0])
t_end = 4 * np.pi  # Two periods

print("RK4 Fixed-Step Integration Demonstration")
print("System: Simple Harmonic Oscillator (ω=1.0)")
print(f"Integration time: 0 to {t_end:.2f} (2 periods)\n")

# Test different step sizes
step_sizes = [0.5, 0.2, 0.1, 0.05]

for dt in step_sizes:
    config = bh.IntegratorConfig.fixed_step(step_size=dt)
    integrator = bh.RK4Integrator(2, dynamics, config=config)

    t = 0.0
    state = state0.copy()
    steps = 0

    # Integrate to end
    while t < t_end - 1e-10:
        state = integrator.step(t, state, dt)
        t += dt
        steps += 1

    # Compare with analytical solution
    exact = analytical(t)
    error = np.linalg.norm(state - exact)

    print(f"Step size dt={dt:5.2f}:")
    print(f"  Steps:      {steps}")
    print(f"  Final state: [{state[0]:.6f}, {state[1]:.6f}]")
    print(f"  Exact:       [{exact[0]:.6f}, {exact[1]:.6f}]")
    print(f"  Error:       {error:.2e}")
    print()

# Expected Output:
# RK4 Fixed-Step Integration Demonstration
# System: Simple Harmonic Oscillator (ω=1.0)
# Integration time: 0 to 12.57 (2 periods)

# Step size dt= 0.50:
#   Steps:      26
#   Final state: [0.907541, -0.413422]
#   Exact:       [0.907447, -0.420167]
#   Error:       6.75e-03

# Step size dt= 0.20:
#   Steps:      63
#   Final state: [0.999412, -0.033457]
#   Exact:       [0.999435, -0.033623]
#   Error:       1.68e-04

# Step size dt= 0.10:
#   Steps:      126
#   Final state: [0.999434, -0.033613]
#   Exact:       [0.999435, -0.033623]
#   Error:       1.05e-05

# Step size dt= 0.05:
#   Steps:      252
#   Final state: [0.999435, -0.033622]
#   Exact:       [0.999435, -0.033623]
#   Error:       6.56e-07
