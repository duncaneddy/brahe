# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using NumericalPropagator with control_input.
Demonstrates adding damping control to a simple harmonic oscillator via control_input.
"""

import numpy as np
import brahe as bh

# Initialize EOP data (needed for epoch operations)
bh.initialize_eop()

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Simple Harmonic Oscillator with damping control
# State: [x, v] where x is position and v is velocity
# Natural dynamics: dx/dt = v, dv/dt = -omega^2 * x
# Control adds damping: u = -c * v
omega = 2.0 * np.pi  # 1 Hz natural frequency
damping_ratio = 0.1  # Damping ratio (zeta)
damping_coeff = 2 * damping_ratio * omega  # c = 2*zeta*omega

# Initial state: displaced from equilibrium
x0 = 1.0  # 1 meter displacement
v0 = 0.0  # Starting from rest
initial_state = np.array([x0, v0])


def sho_dynamics(t, state, params):
    """Simple harmonic oscillator dynamics (undamped).

    This function defines only the natural dynamics.
    Control is added separately via control_input.
    """
    x, v = state[0], state[1]
    omega_sq = params[0] if params is not None else omega**2
    return np.array([v, -omega_sq * x])


def damping_control(t, state, params):
    """Damping control input: u = -c * v (opposes velocity).

    The control_input function returns a state derivative contribution
    that is added to the dynamics output at each integration step.
    """
    dx = np.zeros(len(state))
    v = state[1]
    # Control adds acceleration that opposes velocity
    dx[1] = -damping_coeff * v
    return dx


# Parameters (omega^2)
params = np.array([omega**2])

# Create propagator with dynamics AND control_input
prop_damped = bh.NumericalPropagator(
    epoch,
    initial_state,
    sho_dynamics,
    bh.NumericalPropagationConfig.default(),
    params,
    control_input=damping_control,  # Separate control function
)

# Create undamped propagator for comparison (no control_input)
prop_undamped = bh.NumericalPropagator(
    epoch,
    initial_state,
    sho_dynamics,
    bh.NumericalPropagationConfig.default(),
    params,
)

# Propagate for several periods
period = 2 * np.pi / omega  # Period = 1 second
prop_damped.propagate_to(epoch + 10 * period)
prop_undamped.propagate_to(epoch + 10 * period)

# Sample trajectory and compare
print("Damped vs Undamped Harmonic Oscillator:")
print(f"  Natural frequency: {omega / (2 * np.pi):.1f} Hz")
print(f"  Damping ratio: {damping_ratio}")
print(f"  Damping coefficient: {damping_coeff:.3f} /s")
print("\nTime (s)  Damped x    Undamped x  Amplitude ratio")
print("-" * 55)

for i in range(11):
    t = i * period  # Sample at period intervals
    state_damped = prop_damped.state(epoch + t)
    state_undamped = prop_undamped.state(epoch + t)
    ratio = abs(state_damped[0]) / max(abs(state_undamped[0]), 1e-10)
    print(
        f"  {t:.1f}       {state_damped[0]:+.4f}      {state_undamped[0]:+.4f}       {ratio:.3f}"
    )

# Validate - damped oscillator should decay
final_damped = prop_damped.state(epoch + 10 * period)
final_undamped = prop_undamped.state(epoch + 10 * period)

# Expected decay: amplitude ~ exp(-zeta*omega*t) = exp(-0.1 * 2*pi * 10) ~ 0.002
expected_ratio = np.exp(-damping_ratio * omega * 10 * period)
actual_ratio = abs(final_damped[0]) / abs(x0)

print("\nAfter 10 periods:")
print(f"  Damped amplitude: {abs(final_damped[0]):.4f} m")
print(f"  Undamped amplitude: {abs(final_undamped[0]):.4f} m")
print(f"  Expected decay ratio: {expected_ratio:.4f}")
print(f"  Actual decay ratio: {actual_ratio:.4f}")

assert abs(final_damped[0]) < abs(final_undamped[0])  # Damped has smaller amplitude
assert actual_ratio < 0.1  # Should decay significantly

print("\nExample validated successfully!")
