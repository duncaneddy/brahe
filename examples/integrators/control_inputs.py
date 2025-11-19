# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Control input example.

Demonstrates using the control input parameter to add external forcing
to dynamics, such as thrust maneuvers.
"""

import brahe as bh
import numpy as np


def dynamics(t, state):
    """Orbital dynamics (gravity only).

    Args:
        t: Time
        state: [x, y, z, vx, vy, vz]
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


# Initial LEO state (500 km altitude, circular orbit)
sma = bh.R_EARTH + 500e3
state_initial = np.array([sma, 0.0, 0.0, 0.0, 7612.6, 0.0])

# Orbital period
period = bh.orbital_period(sma)

# Create integrator WITH control input parameter
config = bh.IntegratorConfig.adaptive(abs_tol=1e-10, rel_tol=1e-8)
integrator_thrust = bh.DP54Integrator(
    6, dynamics, jacobian=None, control_fn=control_input, config=config
)

# Create integrator without control (for comparison)
integrator_coast = bh.DP54Integrator(6, dynamics, config=config)

# Propagate with thrust for one orbit
state_thrust = state_initial.copy()
t = 0.0
dt = 60.0

while t < period:
    result = integrator_thrust.step(t, state_thrust, dt)
    state_thrust = result.state
    t += result.dt_used
    dt = result.dt_next

# Propagate without thrust for comparison
state_coast = state_initial.copy()
t = 0.0
dt = 60.0

while t < period:
    result = integrator_coast.step(t, state_coast, dt)
    state_coast = result.state
    t += result.dt_used
    dt = result.dt_next

# Results
r_initial = np.linalg.norm(state_initial[:3])
r_thrust = np.linalg.norm(state_thrust[:3])
r_coast = np.linalg.norm(state_coast[:3])

print(f"Initial radius: {r_initial / 1000:.3f} km")
print(f"Orbital period: {period / 3600:.2f} hours")
print("\nAfter one orbit:")
print(
    f"  With thrust: {r_thrust / 1000:.3f} km (delta_r = {(r_thrust - r_initial) / 1000:.3f} km)"
)
print(
    f"  Coast only:  {r_coast / 1000:.3f} km (delta_r = {(r_coast - r_initial) / 1000:.3f} km)"
)
# Initial radius: 6878.136 km
# Orbital period: 1.58 hours

# After one orbit:
#   With thrust: 6889.325 km (delta_r = 11.189 km)
#   Coast only:  6878.136 km (delta_r = 0.000 km)
