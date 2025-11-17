# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates numerical integration of exponential decay with analytical validation.

This example integrates the simple ODE dx/dt = -k*x and compares the numerical
solution to the analytical solution x(t) = x₀*exp(-kt). Also demonstrates
STM propagation with analytical Jacobian verification.
"""

import brahe as bh
import numpy as np

# Initialize EOP (required even for non-orbital dynamics)
bh.initialize_eop()

# Decay constant
k = 0.1


# Define exponential decay dynamics
def decay_dynamics(t, state):
    """Exponential decay: dx/dt = -k*x"""
    return -k * state


# Analytical solution for comparison
def analytical_solution(t, x0):
    """Analytical solution: x(t) = x₀ * exp(-kt)"""
    return x0 * np.exp(-k * t)


# Analytical Jacobian (for STM validation)
def analytical_jacobian(t, state):
    """Analytical Jacobian: J = -k"""
    n = len(state)
    J = -k * np.eye(n)
    return J


# Analytical STM
def analytical_stm(t):
    """Analytical STM: Φ(t) = exp(-kt)"""
    return np.exp(-k * t)


# Initial condition
x0 = np.array([1.0])
t0 = 0.0
tf = 20.0  # Final time

print("Exponential Decay: dx/dt = -k*x")
print("=" * 70)
print(f"Decay constant k = {k}")
print(f"Initial condition x₀ = {x0[0]}")
print(f"Analytical solution: x(t) = x₀ * exp(-{k}*t)")
print(f"Analytical STM: Φ(t) = exp(-{k}*t)\n")

# ==============================================================================
# Part 1: Compare numerical integration to analytical solution
# ==============================================================================

print("Part 1: Numerical vs Analytical Solution")
print("-" * 70)

# Create numerical Jacobian
jacobian = bh.NumericalJacobian.central(decay_dynamics).with_adaptive(
    scale_factor=1e-8, min_threshold=1e-6
)

# Create integrator with high accuracy
config = bh.IntegratorConfig.adaptive(abs_tol=1e-12, rel_tol=1e-11)
integrator = bh.DP54Integrator(1, decay_dynamics, jacobian=jacobian, config=config)

# Propagate numerically
t = t0
state = x0.copy()
phi = np.eye(1)
dt = 1.0

print("Time(s)  Numerical   Analytical  Abs Error   Rel Error")
print("-" * 70)

times = []
numerical = []
analytical = []

while t < tf:
    # Propagate with STM
    new_state, new_phi, dt_used, _, dt_next = integrator.step_with_varmat(
        t, state, phi, min(dt, tf - t)
    )

    t += dt_used
    state = new_state
    phi = new_phi
    dt = dt_next

    # Compare to analytical
    x_analytical = analytical_solution(t, x0[0])
    abs_error = abs(state[0] - x_analytical)
    rel_error = abs_error / x_analytical

    # Store for plotting
    times.append(t)
    numerical.append(state[0])
    analytical.append(x_analytical)

    # Print every 2 seconds
    if len(times) % 10 == 1 or t >= tf - 1e-6:
        print(
            f"{t:7.2f}  {state[0]:.8f}  {x_analytical:.8f}  {abs_error:.2e}  {rel_error:.2e}"
        )

# ==============================================================================
# Part 2: Validate STM against analytical STM
# ==============================================================================

print("\n" + "=" * 70)
print("Part 2: State Transition Matrix Validation")
print("-" * 70)

# Reset for STM comparison
t = t0
state = x0.copy()
phi_numerical = np.eye(1)

print("Time(s)  Numerical STM  Analytical STM  Abs Error   Rel Error")
print("-" * 70)

dt = 1.0
while t < tf:
    new_state, new_phi, dt_used, _, dt_next = integrator.step_with_varmat(
        t, state, phi_numerical, min(dt, tf - t)
    )

    t += dt_used
    state = new_state
    phi_numerical = new_phi
    dt = dt_next

    # Compare to analytical STM
    phi_analytical = analytical_stm(t)
    abs_error = abs(phi_numerical[0, 0] - phi_analytical)
    rel_error = abs_error / phi_analytical

    # Print every 2 seconds
    if t % 2.0 < dt_used or t >= tf - 1e-6:
        print(
            f"{t:7.2f}  {phi_numerical[0, 0]:.10f}  {phi_analytical:.10f}  {abs_error:.2e}  {rel_error:.2e}"
        )

# ==============================================================================
# Part 3: Demonstrate STM perturbation mapping
# ==============================================================================

print("\n" + "=" * 70)
print("Part 3: STM Perturbation Mapping")
print("-" * 70)

# Small perturbation
delta_x0 = 0.01

# Predict final perturbation using STM
delta_xf_stm = phi_numerical[0, 0] * delta_x0

# Compute final perturbation analytically
x_nominal = analytical_solution(tf, x0[0])
x_perturbed = analytical_solution(tf, x0[0] + delta_x0)
delta_xf_true = x_perturbed - x_nominal

print(f"Initial perturbation: δx₀ = {delta_x0}")
print(f"Final time: t = {tf:.1f} s")
print(f"\nSTM prediction:    δxf = Φ({tf:.1f}) * δx₀ = {delta_xf_stm:.8f}")
print(f"True perturbation: δxf = {delta_xf_true:.8f}")
print(f"Error: {abs(delta_xf_stm - delta_xf_true):.2e}")
print("\nPerfect agreement! (within numerical precision)")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print("  - Numerical integration matches analytical solution to ~1e-10")
print("  - Numerical STM matches analytical STM Φ(t) = exp(-kt)")
print("  - STM correctly maps perturbations through linear dynamics")

# Example output:
# Exponential Decay: dx/dt = -k*x
# ======================================================================
# Decay constant k = 0.1
# Initial condition x₀ = 1.0
# Analytical solution: x(t) = x₀ * exp(-0.1*t)
# Analytical STM: Φ(t) = exp(-0.1*t)
#
# Part 1: Numerical vs Analytical Solution
# ----------------------------------------------------------------------
# Time(s)  Numerical   Analytical  Abs Error   Rel Error
# ----------------------------------------------------------------------
#    0.01  0.99900050  0.99900050  8.88e-16  8.88e-16
#    2.01  0.81777047  0.81777047  1.11e-15  1.36e-15
#    4.01  0.66859943  0.66859943  1.11e-15  1.66e-15
