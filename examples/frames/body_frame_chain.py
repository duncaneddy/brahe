# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Register a spacecraft body frame and a sensor frame mounted on it, then route
the Sun direction through the chain into the sensor frame.
"""

import numpy as np

import brahe as bh

bh.clear_frame_registry()
bh.clear_object_registry()

# A ReferenceFrame::Body or ReferenceFrame::OrbitRelative frame is a pure label until it is
# bound to an object; a ReferenceFrame::Celestial frame (CelestialFrame.GCRF, ...) is
# always bound. The family staticmethods (ReferenceFrame.RTN, ReferenceFrame.SC_BODY,
# ReferenceFrame.CSS, ...) construct a bound frame directly; ReferenceFrame.body(None, ...)
# leaves it unbound.
rtn = bh.ReferenceFrame.RTN("SC")
label = bh.ReferenceFrame.body(None, bh.BodyFrame.SC_BODY())
print(f"{rtn}: bound={rtn.is_bound()}, object={rtn.object()}")
print(f"{label}: bound={label.is_bound()}, object={label.object()}")

# Register "SC" as an object: a callable Epoch -> state (m, m/s) in GCRF. An
# OrbitTrajectory, or an OEM's `register_for` one-liner (see the CCSDS OEM
# docs), registers the same way.
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
x_sc = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
bh.register_object("SC", lambda epc: x_sc, bh.CelestialFrame.GCRF)

# Register SC's body frame and a coarse sun sensor mounted on it. A constant
# attitude (Quaternion, RotationMatrix, EulerAngle, EulerAxis) registers
# directly; orientation chains driven by an attitude ephemeris ship in a
# later release.
q_body = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
q_css = bh.EulerAxis(
    np.array([0.0, 1.0, 0.0]), 0.7, bh.AngleFormat.RADIANS
).to_quaternion()
bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, q_body)
bh.register_frame(
    bh.ReferenceFrame.CSS("SC", "1"), bh.ReferenceFrame.SC_BODY("SC"), q_css
)

# Route the Sun's GCRF position through GCRF -> SC_BODY -> CSS_1.
epc = bh.Epoch.from_date(2024, 3, 1, bh.TimeSystem.UTC)
sun_gcrf = bh.sun_position(epc)
sun_css = bh.position_frame_to_frame(
    bh.CelestialFrame.GCRF, bh.ReferenceFrame.CSS("SC", "1"), epc, sun_gcrf
)
print(f"\nSun direction in CSS_1: {sun_css}")

# Body frames share their object's origin exactly: routing SC's own position
# into CSS_1 lands at the origin, with no lever arm applied.
sc_in_css = bh.position_frame_to_frame(
    bh.CelestialFrame.GCRF, bh.ReferenceFrame.CSS("SC", "1"), epc, x_sc[:3]
)
print(f"SC origin in CSS_1 (zero lever arm): {sc_in_css}")
np.testing.assert_allclose(sc_in_css, 0.0, atol=1e-6)

# Querying an unregistered link raises with a fix: which frame is missing and
# the register_frame call that would supply it.
try:
    bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.CSS("SC", "2"), epc
    )
except RuntimeError as e:
    print(f"\nMissing-link error: {e}")

bh.clear_frame_registry()
bh.clear_object_registry()
print("\nExample validated successfully!")
