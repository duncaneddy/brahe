# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Sensitivity matrix computation example.

Demonstrates using NumericalSensitivity and AnalyticSensitivity providers
to compute ∂f/∂p (sensitivity of dynamics with respect to consider parameters).
"""

import brahe as bh
import numpy as np


def dynamics_with_params(t, state, params):
    """Orbital dynamics with consider parameters.

    Args:
        t: Time
        state: [x, y, z, vx, vy, vz]
        params: [cd_area_m] - drag coefficient * area / mass
    """
    # Extract parameter
    cd_area_m = params[0]

    # Gravitational dynamics
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


def analytical_sensitivity(t, state, params):
    """Analytical sensitivity ∂f/∂p for drag parameter.

    Args:
        t: Time
        state: [x, y, z, vx, vy, vz]
        params: [cd_area_m]

    Returns:
        6x1 sensitivity matrix
    """
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)

    # Atmospheric density
    h = r_norm - bh.R_EARTH
    rho0 = 1.225
    H = 8500.0
    rho = rho0 * np.exp(-h / H)

    v_norm = np.linalg.norm(v)

    # ∂(state_dot)/∂(cd_area_m)
    sens = np.zeros((6, 1))
    if v_norm > 0:
        # ∂(a_drag)/∂(cd_area_m) = -0.5 * rho * v_norm * v
        sens[3:6, 0] = -0.5 * rho * v_norm * v

    return sens


# Initial state (400 km LEO circular orbit)
oe = np.array([bh.R_EARTH + 250e3, 0.001, 51.6, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Consider parameters
params = np.array([0.044])  # cd_area_m = Cd*A/m = 2.2*10/500

# Create numerical sensitivity provider (use class directly as constructor)
numerical_sens = bh.NumericalSensitivity(dynamics_with_params)

# Compute sensitivity matrix numerically
sens_numerical = numerical_sens.compute(0.0, state, params)

print("Numerical sensitivity (∂f/∂p):")
print(
    f"  Position rates: [{sens_numerical[0, 0]}, {sens_numerical[1, 0]}, {sens_numerical[2, 0]}]"
)
print(
    f"  Velocity rates: [{sens_numerical[3, 0]}, {sens_numerical[4, 0]}, {sens_numerical[5, 0]}]"
)

# Create analytical sensitivity provider
analytic_sens = bh.AnalyticSensitivity(analytical_sensitivity)

# Compute sensitivity matrix analytically
sens_analytical = analytic_sens.compute(0.0, state, params)

print("\nAnalytical sensitivity (∂f/∂p):")
print(
    f"  Position rates: [{sens_analytical[0, 0]}, {sens_analytical[1, 0]}, {sens_analytical[2, 0]}]"
)
print(
    f"  Velocity rates: [{sens_analytical[3, 0]}, {sens_analytical[4, 0]}, {sens_analytical[5, 0]}]"
)

# Compare numerical and analytical
diff = np.abs(sens_numerical - sens_analytical)
print(f"\nMax difference: {np.max(diff):.3e}")

# Numerical sensitivity (∂f/∂p):
#   Position rates: [0, 0, 0]
#   Velocity rates: [0, -0.000008425648220011794, -0.000010630522385923769]

# Analytical sensitivity (∂f/∂p):
#   Position rates: [0, 0, 0]
#   Velocity rates: [0, -0.000008425648218908737, -0.00001063052238539645]

# Max difference: 1.103e-15
