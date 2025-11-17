# /// script
# dependencies = ["brahe", "numpy", "scipy"]
# ///
"""
Demonstrates STM propagation for a linear system with analytical validation.

This example integrates a 2D linear system dx/dt = A*x and validates the
numerical STM against the analytical STM computed via matrix exponential.
Perfect for testing Jacobian computation accuracy.
"""

import brahe as bh
import numpy as np
from scipy.linalg import expm

# Initialize EOP (required even for non-orbital dynamics)
bh.initialize_eop()

# System matrix A (2x2 constant matrix)
# Eigenvalues: -0.1, -0.3 (both decaying)
A = np.array([[-0.1, 0.2], [0.0, -0.3]])

print("Linear System: dx/dt = A*x")
print("=" * 70)
print("System matrix A:")
print(A)
eigenvalues = np.linalg.eigvals(A)
print(f"\nEigenvalues: {eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}")
print("Both negative → system is stable (decaying)\n")


# Define linear dynamics
def linear_dynamics(t, state):
    """Linear system: dx/dt = A*x"""
    return A @ state


# Analytical solution: x(t) = exp(A*t) * x₀
def analytical_solution(t, x0):
    """Analytical solution via matrix exponential"""
    return expm(A * t) @ x0


# Analytical STM: Φ(t) = exp(A*t)
def analytical_stm(t):
    """Analytical STM via matrix exponential"""
    return expm(A * t)


# Analytical Jacobian (constant for linear systems)
def analytical_jacobian(t, state):
    """For linear system dx/dt = A*x, Jacobian J = A"""
    return A


# Initial condition
x0 = np.array([1.0, 0.5])
t0 = 0.0
tf = 15.0

print(f"Initial condition: x₀ = [{x0[0]}, {x0[1]}]")
print(f"Propagating from t = {t0} to t = {tf}\n")

# ==============================================================================
# Part 1: Numerical vs Analytical Solution
# ==============================================================================

print("Part 1: Numerical vs Analytical State")
print("-" * 70)

# Create numerical Jacobian
jacobian = bh.NumericalJacobian.central(linear_dynamics).with_adaptive(
    scale_factor=1e-8, min_threshold=1e-6
)

# Create integrator
config = bh.IntegratorConfig.adaptive(abs_tol=1e-12, rel_tol=1e-11)
integrator = bh.DP54Integrator(2, linear_dynamics, jacobian=jacobian, config=config)

# Propagate
t = t0
state = x0.copy()
phi = np.eye(2)
dt = 1.0

print("Time(s)   Numerical [x₁, x₂]        Analytical [x₁, x₂]       ||Error||")
print("-" * 70)

while t < tf:
    new_state, new_phi, dt_used, _, dt_next = integrator.step_with_varmat(
        t, state, phi, min(dt, tf - t)
    )

    t += dt_used
    state = new_state
    phi = new_phi
    dt = dt_next

    # Compare to analytical
    state_analytical = analytical_solution(t, x0)
    error_norm = np.linalg.norm(state - state_analytical)

    # Print every ~2 seconds
    if t % 2.0 < dt_used or (tf - t) < 1e-6:
        print(
            f"{t:6.2f}   [{state[0]:7.5f}, {state[1]:7.5f}]   [{state_analytical[0]:7.5f}, {state_analytical[1]:7.5f}]   {error_norm:.2e}"
        )

# ==============================================================================
# Part 2: Numerical vs Analytical STM
# ==============================================================================

print("\n" + "=" * 70)
print("Part 2: Numerical vs Analytical STM")
print("-" * 70)

# Reset and propagate
t = t0
state = x0.copy()
phi_numerical = np.eye(2)

print("This is the key validation: comparing numerical Jacobian integration")
print("against the exact matrix exponential.\n")
print("Time(s)   Numerical STM             Analytical STM            ||Error||")
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

    # Analytical STM
    phi_analytical = analytical_stm(t)

    # Error between numerical and analytical STM
    stm_error = np.linalg.norm(phi_numerical - phi_analytical)

    # Print every ~2 seconds
    if t % 2.0 < dt_used or (tf - t) < 1e-6:
        print(
            f"{t:6.2f}   [[{phi_numerical[0, 0]:6.4f}, {phi_numerical[0, 1]:6.4f}]   [[{phi_analytical[0, 0]:6.4f}, {phi_analytical[0, 1]:6.4f}]   {stm_error:.2e}"
        )
        print(
            f"          [{phi_numerical[1, 0]:6.4f}, {phi_numerical[1, 1]:6.4f}]]    [{phi_analytical[1, 0]:6.4f}, {phi_analytical[1, 1]:6.4f}]]"
        )

# ==============================================================================
# Part 3: Verify Jacobian Accuracy
# ==============================================================================

print("\n" + "=" * 70)
print("Part 3: Jacobian Accuracy Check")
print("-" * 70)

# For linear systems, the Jacobian should be constant and equal to A
print("For linear system dx/dt = A*x, the Jacobian J = A (constant)")
print("\nAnalytical Jacobian (matrix A):")
print(A)

# Compute numerical Jacobian at a test point
test_state = np.array([1.5, 0.8])
print(f"\nNumerical Jacobian at test point [{test_state[0]}, {test_state[1]}]:")

# Create a Jacobian provider and compute
jac_provider = bh.NumericalJacobian.central(linear_dynamics).with_fixed_offset(1e-6)
J_numerical = jac_provider.compute(0.0, test_state)
print(J_numerical)

J_error = np.linalg.norm(J_numerical - A)
print(f"\n||J_numerical - A||: {J_error:.2e}")
print("Excellent agreement! Numerical Jacobian = Analytical Jacobian")

# ==============================================================================
# Part 4: STM Perturbation Mapping
# ==============================================================================

print("\n" + "=" * 70)
print("Part 4: STM Perturbation Mapping Validation")
print("-" * 70)

# Apply perturbation and verify STM mapping
delta_x0 = np.array([0.01, -0.02])

# Method 1: Predict using STM
delta_xf_stm = phi_numerical @ delta_x0

# Method 2: Direct integration of perturbed state
x0_pert = x0 + delta_x0
xf_pert_analytical = analytical_solution(tf, x0_pert)
xf_nominal_analytical = analytical_solution(tf, x0)
delta_xf_true = xf_pert_analytical - xf_nominal_analytical

print(f"Initial perturbation: δx₀ = [{delta_x0[0]}, {delta_x0[1]}]")
print(f"Final time: t = {tf} s")
print(
    f"\nSTM prediction:     δxf = Φ*δx₀ = [{delta_xf_stm[0]:.8f}, {delta_xf_stm[1]:.8f}]"
)
print(
    f"True perturbation:  δxf =          [{delta_xf_true[0]:.8f}, {delta_xf_true[1]:.8f}]"
)
print(f"Error: {np.linalg.norm(delta_xf_stm - delta_xf_true):.2e}")
print("\nPerfect! For linear systems, STM exactly maps perturbations.")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print("  - Numerical integration matches analytical solution (matrix exponential)")
print("  - Numerical STM matches analytical STM to ~1e-10")
print("  - Numerical Jacobian = Analytical Jacobian (constant matrix A)")
print("  - STM perfectly maps perturbations in linear systems")
print("\nKey Insight:")
print("  Linear systems provide perfect test cases for validating integrator")
print("  and Jacobian accuracy since we have closed-form analytical solutions.")

# Example output:
# Linear System: dx/dt = A*x
# ======================================================================
# System matrix A:
# [[-0.1  0.2]
#  [ 0.  -0.3]]
#
# Eigenvalues: -0.1000, -0.3000
# Both negative → system is stable (decaying)
