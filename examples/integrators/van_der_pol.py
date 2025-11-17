# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates integration of the stiff Van der Pol oscillator.

The Van der Pol oscillator is a nonlinear system with limit cycle behavior.
For large μ, it becomes stiff, challenging adaptive step control. This example
shows how different tolerances affect integration performance and accuracy.
"""

import brahe as bh
import numpy as np

# Initialize EOP (required even for non-orbital dynamics)
bh.initialize_eop()

# Van der Pol parameter (controls stiffness)
# μ = 1.0  → mildly nonlinear
# μ = 5.0  → moderately stiff
# μ = 10.0 → very stiff
mu = 5.0


# Define Van der Pol oscillator dynamics
def van_der_pol(t, state):
    """
    Van der Pol oscillator: ẍ - μ(1 - x²)ẋ + x = 0
    As a system: ẋ₁ = x₂
                 ẋ₂ = μ(1 - x₁²)x₂ - x₁
    """
    x1, x2 = state
    dx1 = x2
    dx2 = mu * (1.0 - x1**2) * x2 - x1
    return np.array([dx1, dx2])


# Initial conditions (standard Van der Pol initialization)
x0 = 2.0  # Initial position (away from equilibrium)
v0 = 0.0  # Initial velocity
state0 = np.array([x0, v0])

# Time parameters
t0 = 0.0
tf = 30.0  # Propagate long enough to see limit cycle

print("Van der Pol Oscillator: ẍ - μ(1 - x²)ẋ + x = 0")
print("=" * 70)
print(f"Stiffness parameter μ = {mu}")
if mu < 2.0:
    print("  → Mildly nonlinear (easy to integrate)")
elif mu < 7.0:
    print("  → Moderately stiff (challenges some integrators)")
else:
    print("  → Very stiff (requires tight tolerances)")
print(f"\nInitial conditions: x₀ = {x0}, v₀ = {v0}")
print("System exhibits limit cycle oscillation\n")

# ==============================================================================
# Part 1: Integration with Appropriate Tolerances
# ==============================================================================

print("Part 1: Integration with High Accuracy Tolerances")
print("-" * 70)

# Create numerical Jacobian
jacobian = bh.NumericalJacobian.central(van_der_pol).with_adaptive(
    scale_factor=1e-8, min_threshold=1e-6
)

# Use tight tolerances for stiff system
config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-8)
integrator = bh.DP54Integrator(2, van_der_pol, jacobian=jacobian, config=config)

# Propagate and track statistics
t = t0
state = state0.copy()
phi = np.eye(2)
dt = 0.1

total_steps = 0
total_time_integrated = 0.0
min_dt = float("inf")
max_dt = 0.0

print("Time(s)   x₁        x₂        dt_used    Steps")
print("-" * 70)

positions = [state[0]]
velocities = [state[1]]
times = [t]

while t < tf:
    new_state, new_phi, dt_used, _, dt_next = integrator.step_with_varmat(
        t, state, phi, min(dt, tf - t)
    )

    t += dt_used
    state = new_state
    phi = new_phi
    dt = dt_next

    total_steps += 1
    total_time_integrated += dt_used
    min_dt = min(min_dt, dt_used)
    max_dt = max(max_dt, dt_used)

    positions.append(state[0])
    velocities.append(state[1])
    times.append(t)

    # Print every ~2 seconds
    if t % 2.0 < dt_used or (tf - t) < 1e-6:
        print(
            f"{t:7.2f}   {state[0]:8.5f}  {state[1]:8.5f}   {dt_used:9.6f}   {total_steps:5d}"
        )

print("\nIntegration Statistics:")
print(f"  Total steps: {total_steps}")
print(f"  Average step size: {total_time_integrated / total_steps:.6f} s")
print(f"  Min step size: {min_dt:.6f} s")
print(f"  Max step size: {max_dt:.6f} s")
print(f"  Step size ratio (max/min): {max_dt / min_dt:.1f}")

# ==============================================================================
# Part 2: Effect of Tolerance on Performance
# ==============================================================================

print("\n" + "=" * 70)
print("Part 2: Effect of Tolerance on Integration Performance")
print("-" * 70)

# Test different tolerance levels
tolerances = [
    (1e-6, 1e-5, "Loose"),
    (1e-9, 1e-8, "Tight"),
    (1e-12, 1e-11, "Very Tight"),
]

print(f"{'Tolerance':15s}  {'Steps':>7s}  {'Avg dt(s)':>10s}  {'Final x₁':>10s}")
print("-" * 70)

for abs_tol, rel_tol, label in tolerances:
    config = bh.IntegratorConfig.adaptive(abs_tol=abs_tol, rel_tol=rel_tol)
    integrator = bh.DP54Integrator(2, van_der_pol, jacobian=jacobian, config=config)

    t = t0
    state = state0.copy()
    phi = np.eye(2)
    dt = 0.1
    steps = 0

    while t < tf:
        new_state, new_phi, dt_used, _, dt_next = integrator.step_with_varmat(
            t, state, phi, min(dt, tf - t)
        )
        t += dt_used
        state = new_state
        phi = new_phi
        dt = dt_next
        steps += 1

    avg_dt = tf / steps
    print(f"{label:15s}  {steps:7d}  {avg_dt:10.6f}  {state[0]:10.5f}")

# ==============================================================================
# Part 3: Phase Space Analysis
# ==============================================================================

print("\n" + "=" * 70)
print("Part 3: Phase Space and Limit Cycle")
print("-" * 70)

# Find approximate limit cycle bounds
x1_min = min(positions)
x1_max = max(positions)
x2_min = min(velocities)
x2_max = max(velocities)

print("Van der Pol oscillator converges to a limit cycle in phase space")
print(f"\nPhase space bounds after {tf:.0f} seconds:")
print(f"  x₁ range: [{x1_min:.4f}, {x1_max:.4f}]")
print(f"  x₂ range: [{x2_min:.4f}, {x2_max:.4f}]")

# Estimate period from zero crossings (approximate)
zero_crossings = []
for i in range(1, len(positions)):
    if positions[i - 1] * positions[i] < 0:  # Sign change
        # Linear interpolation to find crossing time
        t_cross = times[i - 1] + (times[i] - times[i - 1]) * abs(positions[i - 1]) / (
            abs(positions[i - 1]) + abs(positions[i])
        )
        zero_crossings.append(t_cross)

if len(zero_crossings) >= 3:
    # Period is twice the interval between adjacent crossings (up and down)
    periods = [
        zero_crossings[i + 1] - zero_crossings[i]
        for i in range(0, len(zero_crossings) - 1, 2)
    ]
    avg_period = sum(periods) / len(periods)
    print(f"\nEstimated limit cycle period: {avg_period:.2f} s")
    print(f"Frequency: {1.0 / avg_period:.3f} Hz")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print(
    f"  - Van der Pol oscillator with μ = {mu} is {'stiff' if mu > 5.0 else 'moderately nonlinear'}"
)
print("  - Adaptive step control handles stiffness by reducing step size")
print(f"  - Step size varies by factor of {max_dt / min_dt:.0f}x during integration")
print("  - Tighter tolerances require more steps but improve accuracy")
print("  - System converges to stable limit cycle oscillation")
print("\nKey Insight:")
print("  Stiff systems require adaptive integrators with appropriate tolerances.")
print("  Fixed-step methods would require very small dt everywhere, wasting compute.")

# Example output:
# Van der Pol Oscillator: ẍ - μ(1 - x²)ẋ + x = 0
# ======================================================================
# Stiffness parameter μ = 5.0
#   → Moderately stiff (challenges some integrators)
#
# Initial conditions: x₀ = 2.0, v₀ = 0.0
# System exhibits limit cycle oscillation
#
# Part 1: Integration with High Accuracy Tolerances
# ----------------------------------------------------------------------
# Time(s)   x₁        x₂        dt_used    Steps
# ----------------------------------------------------------------------
#    2.00    0.48526  -2.14387    0.013542     148
#    4.00   -1.86214   0.62445    0.013542     296
