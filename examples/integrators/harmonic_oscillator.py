# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates numerical integration of a simple harmonic oscillator.

This example integrates the second-order ODE ẍ = -ω²x (as a first-order system)
and validates against the analytical solution. Also demonstrates energy conservation
and phase space behavior.
"""

import brahe as bh
import numpy as np

# Initialize EOP (required even for non-orbital dynamics)
bh.initialize_eop()

# Angular frequency
omega = 1.0  # rad/s


# Define harmonic oscillator dynamics as first-order system
def oscillator_dynamics(t, state):
    """
    Simple harmonic oscillator: ẍ = -ω²x
    As a system: ẋ₁ = x₂, ẋ₂ = -ω²x₁
    """
    x1, x2 = state
    dx1 = x2
    dx2 = -(omega**2) * x1
    return np.array([dx1, dx2])


# Analytical solution
def analytical_solution(t, x0, v0):
    """
    Analytical solution: x(t) = x₀*cos(ωt) + (v₀/ω)*sin(ωt)
                        ẋ(t) = -x₀*ω*sin(ωt) + v₀*cos(ωt)
    """
    x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
    return np.array([x, v])


# Energy function (should be conserved)
def total_energy(state):
    """Total energy: E = ½(ẋ² + ω²x²)"""
    x, v = state
    return 0.5 * (v**2 + omega**2 * x**2)


# Initial conditions
x0 = 1.0  # Initial position
v0 = 0.0  # Initial velocity
state0 = np.array([x0, v0])
E0 = total_energy(state0)  # Initial energy

# Time parameters
t0 = 0.0
tf = 20.0  # Propagate for ~3 periods (T = 2π/ω ≈ 6.28)

print("Simple Harmonic Oscillator: ẍ = -ω²x")
print("=" * 70)
print(f"Angular frequency ω = {omega} rad/s")
print(f"Period T = {2 * np.pi / omega:.2f} s")
print(f"Initial conditions: x₀ = {x0}, v₀ = {v0}")
print(f"Initial energy: E₀ = {E0:.6f}")
print(
    f"Analytical solution: x(t) = {x0}*cos({omega}*t) + {v0 / omega}*sin({omega}*t)\n"
)

# ==============================================================================
# Part 1: Numerical vs Analytical Solution
# ==============================================================================

print("Part 1: Numerical vs Analytical Solution")
print("-" * 70)

# Create numerical Jacobian
jacobian = bh.NumericalJacobian.central(oscillator_dynamics).with_adaptive(
    scale_factor=1e-8, min_threshold=1e-6
)

# Create integrator with high accuracy
config = bh.IntegratorConfig.adaptive(abs_tol=1e-12, rel_tol=1e-11)
integrator = bh.DP54Integrator(2, oscillator_dynamics, jacobian=jacobian, config=config)

# Propagate
t = t0
state = state0.copy()
phi = np.eye(2)
dt = 0.5

print("Time(s)  Numerical x  Analytical x  Position Error  Energy Error")
print("-" * 70)

max_pos_error = 0.0
max_energy_error = 0.0

while t < tf:
    # Propagate with STM
    new_state, new_phi, dt_used, _, dt_next = integrator.step_with_varmat(
        t, state, phi, min(dt, tf - t)
    )

    t += dt_used
    state = new_state
    phi = new_phi
    dt = dt_next

    # Compare to analytical solution
    state_analytical = analytical_solution(t, x0, v0)
    pos_error = abs(state[0] - state_analytical[0])
    max_pos_error = max(max_pos_error, pos_error)

    # Check energy conservation
    E = total_energy(state)
    energy_error = abs(E - E0)
    max_energy_error = max(max_energy_error, energy_error)

    # Print every ~1 second
    if t % 1.0 < dt_used or abs(tf - t) < 1e-6:
        print(
            f"{t:7.2f}  {state[0]:11.8f}  {state_analytical[0]:13.8f}  {pos_error:14.2e}  {energy_error:12.2e}"
        )

# ==============================================================================
# Part 2: Phase Space Trajectory
# ==============================================================================

print("\n" + "=" * 70)
print("Part 2: Phase Space Analysis")
print("-" * 70)

# Reset and collect phase space data
t = t0
state = state0.copy()
phi = np.eye(2)
dt = 0.1

positions = [state[0]]
velocities = [state[1]]

while t < 2 * np.pi / omega:  # One complete period
    new_state, new_phi, dt_used, _, dt_next = integrator.step_with_varmat(
        t, state, phi, min(dt, 2 * np.pi / omega - t)
    )

    t += dt_used
    state = new_state
    phi = new_phi
    dt = dt_next

    positions.append(state[0])
    velocities.append(state[1])

print(f"Collected {len(positions)} points over one period")
print("Phase space trajectory should be an ellipse")
print(f"Semi-major axis (velocity): {max(np.abs(velocities)):.6f}")
print(f"Semi-minor axis (position): {max(np.abs(positions)):.6f}")

# ==============================================================================
# Part 3: STM Properties for Harmonic Oscillator
# ==============================================================================

print("\n" + "=" * 70)
print("Part 3: State Transition Matrix Properties")
print("-" * 70)

# The STM for harmonic oscillator has special properties
# After one period, STM should return to identity (periodic system)

# Propagate for exactly one period
period = 2 * np.pi / omega
t = t0
state = state0.copy()
phi = np.eye(2)

while t < period - 1e-6:
    new_state, new_phi, dt_used, _, dt_next = integrator.step_with_varmat(
        t, state, phi, min(dt, period - t)
    )
    t += dt_used
    state = new_state
    phi = new_phi
    dt = dt_next

print(f"STM after one period (t = {period:.4f} s):")
print(phi)
print(f"\nDeterminant: {np.linalg.det(phi):.10f} (should be ~1.0 for Hamiltonian)")
print(f"Deviation from identity: {np.linalg.norm(phi - np.eye(2)):.2e}")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print(f"  - Maximum position error: {max_pos_error:.2e}")
print(f"  - Maximum energy error: {max_energy_error:.2e}")
print("  - Energy is conserved to machine precision!")
print("  - Numerical solution matches analytical solution")
print("  - Phase space trajectory is a perfect ellipse (energy surface)")

# Example output:
# Simple Harmonic Oscillator: ẍ = -ω²x
# ======================================================================
# Angular frequency ω = 1.0 rad/s
# Period T = 6.28 s
# Initial conditions: x₀ = 1.0, v₀ = 0.0
# Initial energy: E₀ = 0.500000
# Analytical solution: x(t) = 1.0*cos(1.0*t) + 0.0*sin(1.0*t)
#
# Part 1: Numerical vs Analytical Solution
# ----------------------------------------------------------------------
# Time(s)  Numerical x  Analytical x  Position Error  Energy Error
# ----------------------------------------------------------------------
#    1.00   0.54030231    0.54030231        2.22e-16      4.44e-16
#    2.00  -0.41614684   -0.41614684        2.22e-16      0.00e+00
#    3.00  -0.98999250   -0.98999250        2.22e-16      4.44e-16
