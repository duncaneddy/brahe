# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates equivalence between STM propagation and direct perturbation integration.

This example shows that for small perturbations, the State Transition Matrix (STM)
can accurately predict the effect of initial perturbations without directly
integrating the perturbed trajectory. This is fundamental for orbit determination
and covariance propagation.
"""

import brahe as bh
import numpy as np

# Initialize EOP
bh.initialize_eop()


# Define two-body orbital dynamics
def dynamics(t, state):
    """Two-body point-mass dynamics with Earth gravity."""
    mu = bh.GM_EARTH
    r = state[0:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    a = -mu / r_norm**3 * r
    return np.concatenate([v, a])


# Create numerical Jacobian for variational equations
jacobian = bh.NumericalJacobian.central(dynamics).with_fixed_offset(0.1)

# Configuration for high accuracy
config = bh.IntegratorConfig.adaptive(abs_tol=1e-12, rel_tol=1e-10)

# Create two integrators:
# 1. With Jacobian - propagates STM alongside state
integrator_nominal = bh.RKN1210Integrator(6, dynamics, jacobian=jacobian, config=config)

# 2. Without Jacobian - for direct perturbation integration
integrator_pert = bh.RKN1210Integrator(6, dynamics, config=config)

# Initial state: circular orbit at 500 km altitude
oe0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0])
state0 = bh.state_osculating_to_cartesian(oe0, bh.AngleFormat.DEGREES)

# Apply 10-meter perturbation in X direction
perturbation = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Integration parameters
total_time = 100.0  # Total propagation time (seconds)
num_steps = 10  # Number of steps
dt = total_time / num_steps

# Initialize states
state_nominal = state0.copy()
phi = np.eye(6)  # State Transition Matrix starts as identity
state_pert = state0 + perturbation

print("STM vs Direct Perturbation Comparison")
print("=" * 70)
print(f"Initial orbit: {(oe0[0] - bh.R_EARTH) / 1e3:.1f} km altitude (circular)")
print(f"Perturbation: {perturbation[0]:.1f} m in X direction")
print(f"Propagating for {total_time:.0f} seconds in {num_steps} steps\n")
print("Theory: For small δx₀, the perturbed state should satisfy:")
print("        x_pert(t) ≈ x_nominal(t) + Φ(t,t₀)·δx₀\n")
print("Step   Time(s)  ||Error||(m)  Max Component(m)  Relative Error")
print("-" * 70)

t = 0.0
for step in range(num_steps):
    # Propagate nominal trajectory with STM
    new_state_nominal, new_phi, dt_used, _, _ = integrator_nominal.step_with_varmat(
        t, state_nominal, phi, dt
    )

    # Propagate perturbed trajectory directly
    result_pert = integrator_pert.step(t, state_pert, dt)

    # Predict perturbed state using STM: x_pert ≈ x_nominal + Φ·δx₀
    state_pert_predicted = new_state_nominal + new_phi @ perturbation

    # Compute error between STM prediction and direct integration
    error = result_pert.state - state_pert_predicted
    error_norm = np.linalg.norm(error)
    error_max = np.max(np.abs(error))

    # Relative error compared to perturbation magnitude
    relative_error = error_norm / np.linalg.norm(perturbation)

    print(
        f"{step + 1:4d}  {t + dt_used:7.1f}  {error_norm:12.6f}  {error_max:16.6f}  {relative_error:13.6f}"
    )

    # Update for next step
    state_nominal = new_state_nominal
    phi = new_phi
    state_pert = result_pert.state
    t += dt_used

# Example output:
# STM vs Direct Perturbation Comparison
# ======================================================================
# Initial orbit: 500.0 km altitude (circular)
# Perturbation: 10.0 m in X direction
# Propagating for 100 seconds in 10 steps
#
# Theory: For small δx₀, the perturbed state should satisfy:
#         x_pert(t) ≈ x_nominal(t) + Φ(t,t₀)·δx₀
#
# Step   Time(s)  ||Error||(m)  Max Component(m)  Relative Error
# ----------------------------------------------------------------------
#    1     10.0      0.000078          0.000053      0.000008
#    2     20.0      0.000299          0.000211      0.000030
#    3     30.0      0.000627          0.000462      0.000063
#    4     40.0      0.001025          0.000791      0.000103
#    5     50.0      0.001463          0.001176      0.000146
#    6     60.0      0.001919          0.001600      0.000192
#    7     70.0      0.002378          0.002057      0.000238
#    8     80.0      0.002831          0.002539      0.000283
#    9     90.0      0.003271          0.003040      0.000327
#   10    100.0      0.003693          0.003556      0.000369
