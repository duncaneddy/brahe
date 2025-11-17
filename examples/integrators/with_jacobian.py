# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrates integration with state transition matrix propagation.

This example shows how to propagate variational equations using a numerical
Jacobian for orbit determination applications.
"""

import brahe as bh
import numpy as np

# Initialize EOP
bh.initialize_eop()


# Define two-body dynamics
def dynamics(t, state):
    mu = bh.GM_EARTH
    r = state[0:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    a = -mu / r_norm**3 * r
    return np.concatenate([v, a])


# Create numerical Jacobian for variational equations
jacobian = bh.NumericalJacobian.central(dynamics).with_adaptive(
    scale_factor=1.0, min_threshold=1e-6
)

# Initial orbit (LEO)
r0 = np.array([bh.R_EARTH + 600e3, 0.0, 0.0])
v0 = np.array([0.0, 7.5e3, 0.0])
state0 = np.concatenate([r0, v0])

# Initial state transition matrix (identity)
phi0 = np.eye(6)

print("Integration with State Transition Matrix")
print(f"Initial orbit: {r0[0] / 1e3:.1f} km altitude")

# Create integrator with Jacobian
config = bh.IntegratorConfig.adaptive(abs_tol=1e-12, rel_tol=1e-11)
integrator = bh.DP54Integrator(6, dynamics, jacobian=jacobian, config=config)

# Propagate for one orbit period
t = 0.0
state = state0.copy()
phi = phi0.copy()
dt = 60.0

# Approximate orbital period
period = bh.orbital_period(np.linalg.norm(r0))

print("Time   Position STM[0,0]  Velocity STM[3,3]  Det(STM)")
print("-" * 60)

steps = 0
while t < period:
    # Propagate state and STM together (adaptive integrator returns 5-tuple)
    new_state, new_phi, dt_used, error_est, dt_next = integrator.step_with_varmat(
        t, state, phi, min(dt, period - t)
    )

    t += dt_used
    state = new_state
    phi = new_phi
    dt = dt_next
    steps += 1

    # Print progress
    if steps % 20 == 1:
        det_phi = np.linalg.det(phi)
        print(
            f"{t:6.0f}s    {phi[0, 0]:8.4f}      {phi[3, 3]:8.4f}        {det_phi:8.4f}"
        )

print(f"\nPropagation complete! ({steps} steps)")

# Example: Map initial position uncertainty to final uncertainty
print("\nExample: Uncertainty Propagation")
dx = 100.0
print(f"Initial position uncertainty: ±{dx} m in each direction")
delta_r0 = np.array([dx, dx, dx, 0.0, 0.0, 0.0])
delta_rf = phi @ delta_r0
print(
    f"Final position uncertainty: [{delta_rf[0]:.1f}, {delta_rf[1]:.1f}, {delta_rf[2]:.1f}] m"
)
print(
    f"Uncertainty growth: {np.linalg.norm(delta_rf[0:3]) / np.linalg.norm(delta_r0[0:3]):.1f}x"
)

# Example output:
# Integration with State Transition Matrix
# Initial orbit: 6978.1 km altitude
# Time   Position STM[0,0]  Velocity STM[3,3]  Det(STM)
# ------------------------------------------------------------
#      0s      1.0000        1.0000          1.0000
#    223s      1.0580        1.0564          1.0000
#    594s      1.3993        1.3227          1.0000
#   1007s      2.0473        1.5071          1.0000
#   1346s      2.5985        1.1989          1.0000
#   1530s      2.8009        0.7891          1.0000
#   1849s      2.7780       -0.2849          1.0000
#   2245s      1.6741       -1.8673          1.0000
#   2608s     -0.7226       -2.9191          1.0000
#   2835s     -2.8726       -3.1249          1.0000
#   3091s     -5.7502       -2.8654          1.0000
#   3455s    -10.0598       -1.7595          1.0000
#   3850s    -13.8989       -0.1764          1.0000
#   4169s    -15.4760        0.8634          1.0000
#   4360s    -15.5400        1.2575          1.0000
#   4700s    -13.8708        1.5097          1.0000
#   5114s     -8.9723        1.2931          1.0000
#   5484s     -2.6152        1.0402          1.0000
#   5697s      1.5156        1.0008          1.0000

# Propagation complete! (370 steps)

# Example: Uncertainty Propagation
# Initial position uncertainty: ±100.0 m in each direction
# Final position uncertainty: [357.4, -1684.0, 99.0] m
# Uncertainty growth: 10.0x
