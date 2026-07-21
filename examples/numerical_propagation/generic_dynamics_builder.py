# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Constructing a NumericalPropagator using the builder API.

The builder takes the three required fields -- epoch, state, and
dynamics_fn -- directly as arguments to builder(). Optional fields such
as params and initial_covariance default when omitted and are set
through chained setters.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

omega = 2.0 * np.pi  # 1 Hz oscillation frequency
initial_state = np.array([1.0, 0.0])  # [x0, v0]


def sho_dynamics(t, state, params):
    """Simple harmonic oscillator dynamics."""
    x, v = state[0], state[1]
    omega_sq = params[0] if params is not None else omega**2
    return np.array([v, -omega_sq * x])


prop = (
    bh.NumericalPropagator.builder(epoch, initial_state, sho_dynamics)
    .propagation_config(bh.NumericalPropagationConfig.default())
    .params(np.array([omega**2]))
    .build()
)

period = 2 * np.pi / omega  # Period = 2*pi/omega = 1 second
prop.propagate_to(epoch + 5 * period)

final_state = prop.current_state()
print(f"Position after 5 periods: {final_state[0]:+.6f} m")
print(f"Velocity after 5 periods: {final_state[1]:+.6f} m/s")

assert abs(final_state[0] - 1.0) < 0.01
print("Example validated successfully!")
