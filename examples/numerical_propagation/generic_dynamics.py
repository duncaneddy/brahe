# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using NumericalPropagator for arbitrary dynamics.
Demonstrates propagating non-orbital systems (simple harmonic oscillator).
"""

import numpy as np
import brahe as bh

# Initialize EOP data (needed for epoch operations)
bh.initialize_eop()

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Simple Harmonic Oscillator (SHO)
# State: [x, v] where x is position and v is velocity
# Dynamics: dx/dt = v, dv/dt = -omega^2 * x
omega = 2.0 * np.pi  # 1 Hz oscillation frequency

# Initial state: displaced from equilibrium
x0 = 1.0  # 1 meter displacement
v0 = 0.0  # Starting from rest
initial_state = np.array([x0, v0])


def sho_dynamics(t, state, params):
    """Simple harmonic oscillator dynamics."""
    x, v = state[0], state[1]
    omega_sq = params[0] if params is not None else omega**2
    return np.array([v, -omega_sq * x])


# Parameters (omega^2)
params = np.array([omega**2])

# Create generic numerical propagator
prop = bh.NumericalPropagator(
    epoch,
    initial_state,
    sho_dynamics,
    bh.NumericalPropagationConfig.default(),
    params,
)

# Propagate for several periods
period = 2 * np.pi / omega  # Period = 2*pi/omega = 1 second
prop.propagate_to(epoch + 5 * period)

# Sample trajectory
print("Simple Harmonic Oscillator Trajectory:")
print("  omega = 2*pi rad/s (1 Hz)")
print("  x0 = 1.0 m, v0 = 0.0 m/s")
print("\nTime (s)  Position (m)  Velocity (m/s)  Analytical x")
print("-" * 55)

for i in range(11):
    t = i * period / 2  # Sample at half-period intervals
    state = prop.state(epoch + t)
    # Analytical solution: x(t) = x0*cos(omega*t), v(t) = -x0*omega*sin(omega*t)
    x_analytical = x0 * np.cos(omega * t)
    print(
        f"  {t:.2f}      {state[0]:+.6f}      {state[1]:+.6f}      {x_analytical:+.6f}"
    )

# Validate - after full period should return to initial
final_state = prop.state(epoch + 5 * period)
error_x = abs(final_state[0] - x0)
error_v = abs(final_state[1] - v0)

print("\nAfter 5 periods:")
print(f"  Position error: {error_x:.2e} m")
print(f"  Velocity error: {error_v:.2e} m/s")

assert error_x < 0.01  # Within 1 cm
assert error_v < 0.1  # Within 10 cm/s

print("\nExample validated successfully!")
