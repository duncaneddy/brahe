# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Derive a body frame's angular velocity numerically from a rotation-only
callback with `numerical_rates_step`, and show the state transform that
requires it.
"""

import numpy as np

import brahe as bh

bh.clear_frame_registry()
bh.clear_object_registry()

t0 = bh.Epoch.from_date(2024, 3, 1, bh.TimeSystem.UTC)
rate = 1.0e-3  # spin rate (rad/s)


def rotation(epc):
    dt = epc - t0
    c, s = np.cos(rate * dt), np.sin(rate * dt)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


# A rotation-only callback carries no angular velocity, so a state transform
# through it fails: the velocity transport term is otherwise undefined.
bh.register_frame(bh.Frame.SC_BODY("SC"), bh.CelestialFrame.GCRF, rotation)
bh.register_object("SC", lambda epc: np.zeros(6), bh.CelestialFrame.GCRF)

epc = t0 + 100.0
x_gcrf = np.array([1.0e3, 2.0e3, 3.0e3, 0.0, 0.0, 0.0])
try:
    bh.state_frame_to_frame(bh.CelestialFrame.GCRF, bh.Frame.SC_BODY("SC"), epc, x_gcrf)
except RuntimeError as e:
    print(f"Rates rule error: {e}")

# Re-registering with `numerical_rates_step` wraps the same callback so a
# missing angular velocity is derived by central differencing the rotation
# over +/- step/2 seconds; a provider that already returns rates is used
# unchanged. The state transform then succeeds.
bh.unregister_frame(bh.Frame.SC_BODY("SC"))
bh.register_frame(
    bh.Frame.SC_BODY("SC"), bh.CelestialFrame.GCRF, rotation, numerical_rates_step=1.0
)
x_body = bh.state_frame_to_frame(
    bh.CelestialFrame.GCRF, bh.Frame.SC_BODY("SC"), epc, x_gcrf
)
print(f"\nBody-frame state with numerical rates: {x_body}")

# Compare against a hand-differenced velocity: with_numerical_rates recovers
# the transport term from the same central-difference recipe.
delta = 0.5
r_plus = rotation(epc + delta) @ x_gcrf[:3]
r_minus = rotation(epc - delta) @ x_gcrf[:3]
v_numerical = (r_plus - r_minus) / (2.0 * delta)
np.testing.assert_allclose(x_body[3:], v_numerical, atol=1e-6)

bh.clear_frame_registry()
bh.clear_object_registry()
print("\nExample validated successfully!")
