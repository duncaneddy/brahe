# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Register a user-defined body-fixed frame from a rotation callback and use it
with the frame router.
"""

import brahe as bh
import numpy as np

# An uncatalogued body (e.g. a newly observed asteroid): self-assign a unique
# negative NAIF ID for its center, mirroring NAIF's convention.
CENTER = -20001

# The frame key is an arbitrary integer handle. It is unrelated to NAIF IDs;
# it only names the callback registered below so BodyFixedCustom can look it
# up at evaluation time.
KEY = 1042

t0 = bh.Epoch.from_date(2024, 3, 1, bh.TimeSystem.TDB)
rate = 2.0e-4  # spin rate (rad/s), ~8.7 h rotation period


# The rotation callback is a proper function: it receives an Epoch and returns
# the 3x3 ICRF -> body-fixed rotation matrix at that instant. Here the body
# spins uniformly about its z-axis.
def rotation(epc):
    theta = rate * (epc - t0)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


# The optional omega callback is also a function of the epoch, returning the
# frame's angular velocity vector (rad/s) for the exact velocity transport
# term. Omitting it falls back to numeric differentiation of `rotation`.
def omega(epc):
    return np.array([0.0, 0.0, rate])


bh.register_custom_frame(KEY, rotation, omega)

inertial = bh.ReferenceFrame.BodyCenteredICRF(CENTER)
fixed = bh.ReferenceFrame.BodyFixedCustom(CENTER, KEY)

# Convert an inertial state about the body into its body-fixed frame. Both
# frames share the same center, so no ephemeris kernel is needed.
epc = t0 + 600.0
x_inertial = np.array([7.0e5, -2.0e5, 3.0e5, 10.0, 25.0, -5.0])
x_fixed = bh.state_frame_to_frame(inertial, fixed, epc, x_inertial)

print(f"Epoch: {epc}")
print(f"Inertial state:   {np.array2string(x_inertial, precision=3)}")
print(f"Body-fixed state: {np.array2string(x_fixed, precision=3)}")

# Round trip back to the inertial frame recovers the input.
x_back = bh.state_frame_to_frame(fixed, inertial, epc, x_fixed)
np.testing.assert_allclose(x_back, x_inertial, atol=1e-6)

# A point co-rotating with the body is stationary in the body-fixed frame.
r_surf = rotation(epc).T @ np.array([1.0e3, 0.0, 0.0])
v_surf = np.cross(omega(epc), r_surf)
x_surf_fixed = bh.state_frame_to_frame(
    inertial, fixed, epc, np.concatenate([r_surf, v_surf])
)
np.testing.assert_allclose(x_surf_fixed[3:], 0.0, atol=1e-9)

bh.unregister_custom_frame(KEY)
print("\nExample validated successfully!")
