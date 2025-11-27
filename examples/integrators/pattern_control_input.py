# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Control input pattern example.

Demonstrates adding a control input function that perturbs the dynamics,
useful for modeling thrust or other external forcing functions.

The control input is passed as a separate parameter to the integrator,
keeping the core dynamics function clean and reusable.
"""

import brahe as bh
import numpy as np


def dynamics(t, state):
    """Orbital dynamics (gravity only).

    Args:
        t: Time
        state: [x, y, z, vx, vy, vz]

    Returns:
        State derivative [vx, vy, vz, ax, ay, az]
    """
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)
    a_grav = -bh.GM_EARTH / (r_norm**3) * r

    return np.concatenate([v, a_grav])


def control_input(t, state):
    """Control input: constant low thrust in velocity direction.

    Args:
        t: Time
        state: [x, y, z, vx, vy, vz]

    Returns:
        Control vector added to state derivative
    """
    v = state[3:]
    v_norm = np.linalg.norm(v)

    control = np.zeros(6)
    if v_norm > 0:
        thrust_magnitude = 0.001  # m/s^2
        control[3:] = thrust_magnitude * v / v_norm

    return control


# Initial LEO state
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Create integrator with control input
# The control function is passed as a separate parameter
config = bh.IntegratorConfig.adaptive(abs_tol=1e-10, rel_tol=1e-8)
integrator = bh.DP54Integrator(
    6, dynamics, jacobian=None, control_fn=control_input, config=config
)

# Integrate for one orbit period
period = bh.orbital_period(oe[0])
t = 0.0
dt = 60.0

print(f"Initial semi-major axis: {oe[0] / 1000:.3f} km")
print(f"Integrating with thrust for {period / 3600:.2f} hours...")

while t < period:
    result = integrator.step(t, state, dt)
    state = result.state
    t += result.dt_used
    dt = result.dt_next

# Check final state
final_oe = bh.state_eci_to_koe(state, bh.AngleFormat.DEGREES)
print(f"Final semi-major axis: {final_oe[0] / 1000:.3f} km")
print(f"Change in SMA: {(final_oe[0] - oe[0]) / 1000:.3f} km")
# Initial semi-major axis: 6878.136 km
# Integrating with thrust for 1.58 hours...
# Final semi-major axis: 6888.420 km
# Change in SMA: 10.283 km
