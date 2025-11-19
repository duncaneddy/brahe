# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Sensitivity matrix propagation pattern example.

Demonstrates propagating the sensitivity matrix alongside state to map
parameter uncertainties to state uncertainties over time.

The sensitivity matrix Φ = ∂x/∂p evolves according to:
    dΦ/dt = (∂f/∂x) * Φ + (∂f/∂p)

This augmented state approach propagates [state, vec(Φ)] together.
"""

import brahe as bh
import numpy as np


def dynamics_with_params(t, state, params):
    """Orbital dynamics with atmospheric drag.

    Args:
        t: Time
        state: [x, y, z, vx, vy, vz]
        params: [cd_area_m] - drag coefficient * area / mass
    """
    cd_area_m = params[0]

    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)
    a_grav = -bh.GM_EARTH / (r_norm**3) * r

    # Atmospheric drag (simplified exponential model)
    h = r_norm - bh.R_EARTH
    rho0 = 1.225  # kg/m^3 at sea level
    H = 8500.0  # Scale height in meters
    rho = rho0 * np.exp(-h / H)

    v_norm = np.linalg.norm(v)
    a_drag = -0.5 * rho * cd_area_m * v_norm * v

    return np.concatenate([v, a_grav + a_drag])


# Consider parameters
cd_area_m = 2.2 * 10.0 / 500.0  # Cd=2.2, A=10m^2, m=500kg
params = np.array([cd_area_m])
num_params = len(params)

# Create sensitivity provider using NumericalSensitivity
sensitivity_provider = bh.NumericalSensitivity.central(dynamics_with_params)

# Create Jacobian provider using NumericalJacobian
jacobian_provider = bh.NumericalJacobian.central(
    lambda t, s: dynamics_with_params(t, s, params)
)


def augmented_dynamics(t, aug_state):
    """Augmented dynamics for state + sensitivity matrix propagation.

    Propagates:
        dx/dt = f(t, x, p)
        dΦ/dt = (∂f/∂x) * Φ + (∂f/∂p)

    Args:
        t: Time
        aug_state: [state (6), vec(Φ) (6*num_params)]
    """
    state = aug_state[:6]
    phi = aug_state[6:].reshape(6, num_params)

    # State derivative
    state_dot = dynamics_with_params(t, state, params)

    # Compute Jacobian ∂f/∂x
    jacobian = jacobian_provider.compute(t, state)

    # Compute sensitivity ∂f/∂p
    sensitivity = sensitivity_provider.compute(t, state, params)

    # Sensitivity matrix derivative: dΦ/dt = J*Φ + S
    phi_dot = jacobian @ phi + sensitivity

    return np.concatenate([state_dot, phi_dot.flatten()])


# Initial state (200 km LEO for significant drag effects)
oe = np.array([bh.R_EARTH + 200e3, 0.001, 51.6, 0.0, 0.0, 0.0])
state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)

# Initial sensitivity matrix (identity would mean we start with unit sensitivity,
# but we start with zero since we're interested in how sensitivity develops)
phi0 = np.zeros((6, num_params))

# Augmented initial state
aug_state = np.concatenate([state, phi0.flatten()])

# Create integrator for augmented system
# Using fixed step RK4 for simplicity and exact parity with Rust
aug_dim = 6 + 6 * num_params
config = bh.IntegratorConfig.fixed_step(1.0)
integrator = bh.RK4Integrator(aug_dim, augmented_dynamics, config=config)

# Propagate for 1 hour
t = 0.0
dt = 1.0
t_final = 3600.0

while t < t_final:
    aug_state = integrator.step(t, aug_state, dt)
    t += dt

# Extract final state and sensitivity matrix
final_state = aug_state[:6]
final_phi = aug_state[6:].reshape(6, num_params)

print(f"Final position after {t_final / 60:.0f} minutes:")
print(f"  x: {final_state[0] / 1000:.3f} km")
print(f"  y: {final_state[1] / 1000:.3f} km")
print(f"  z: {final_state[2] / 1000:.3f} km")

print("\nSensitivity matrix Φ = ∂x/∂p (position per unit Cd*A/m):")
print(f"  dx/dp: {final_phi[0, 0]:.3f} m/(m²/kg)")
print(f"  dy/dp: {final_phi[1, 0]:.3f} m/(m²/kg)")
print(f"  dz/dp: {final_phi[2, 0]:.3f} m/(m²/kg)")

print("\nSensitivity matrix Φ = ∂x/∂p (velocity per unit Cd*A/m):")
print(f"  dvx/dp: {final_phi[3, 0]:.6f} m/s/(m²/kg)")
print(f"  dvy/dp: {final_phi[4, 0]:.6f} m/s/(m²/kg)")
print(f"  dvz/dp: {final_phi[5, 0]:.6f} m/s/(m²/kg)")

# Interpret: If we have 10% uncertainty in Cd*A/m (0.1 * 0.044 = 0.0044),
# the position uncertainty after 1 hour would be:
delta_p = 0.1 * cd_area_m
pos_uncertainty = np.linalg.norm(final_phi[:3, 0]) * delta_p
print(f"\nPosition uncertainty for 10% parameter uncertainty: {pos_uncertainty:.1f} m")

# Expected output:
# Final position after 60 minutes:
#   x: -2884.245 km
#   y: -3673.659 km
#   z: -4635.004 km

# Sensitivity matrix Φ = ∂x/∂p (position per unit Cd*A/m):
#   dx/dp: 59942.894 m/(m²/kg)
#   dy/dp: -3796.878 m/(m²/kg)
#   dz/dp: -4790.467 m/(m²/kg)

# Sensitivity matrix Φ = ∂x/∂p (velocity per unit Cd*A/m):
#   dvx/dp: 44.091413 m/s/(m²/kg)
#   dvy/dp: 33.444231 m/s/(m²/kg)
#   dvz/dp: 42.196119 m/s/(m²/kg)

# Position uncertainty for 10% parameter uncertainty: 265.1 m
