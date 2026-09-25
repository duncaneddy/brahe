# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build the NSW frame on an inclined orbit with a moving Sun state, check the Y axis against the Sun direction, compare the fixed-Sun and moving-Sun rates, do a relative-state round trip, and evaluate it through the frame graph.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

oe = np.array([bh.R_EARTH + 700e3, 0.05, 97.8, 15.0, 30.0, 45.0])
x_sc = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
r_sun = bh.sun_position(epc)
v_sun = (bh.sun_position(epc + 0.5) - bh.sun_position(epc - 0.5)) / 1.0
x_sun = np.concatenate([r_sun, v_sun])

r_nsw = bh.rotation_nsw_to_eci(x_sc, x_sun)
x_axis, y_axis = r_nsw[:, 0], r_nsw[:, 1]
sun_dir = (x_sun[:3] - x_sc[:3]) / np.linalg.norm(x_sun[:3] - x_sc[:3])
print("Y . sun_direction > 0:", y_axis.dot(sun_dir) > 0.0)
print("|Y . X| < 1e-12:", abs(y_axis.dot(x_axis)) < 1e-12)

# Fixed-Sun approximation: a zero-velocity Sun state omits the Sun's own motion
x_sun_fixed = np.concatenate([r_sun, np.zeros(3)])
omega_moving = bh.omega_nsw(x_sc, x_sun)
omega_fixed = bh.omega_nsw(x_sc, x_sun_fixed)
rate_diff = np.linalg.norm(omega_moving - omega_fixed)
print(
    "Fixed-Sun approximation error within (1e-8, 1e-6) rad/s:",
    1e-8 < rate_diff < 1e-6,
)

# Relative state round trip: 1 km along Y, 200 m along Z
x_rel = np.array([0.0, 1000.0, 200.0, 0.0, 0.0, 0.0])
x_deputy = bh.state_nsw_to_eci(x_sc, x_rel, x_sun)
print(
    f"Round trip error: {np.linalg.norm(bh.state_eci_to_nsw(x_sc, x_deputy, x_sun) - x_rel):.2e}"
)

# Frame graph, forcing the analytic Sun model
bh.set_frame_ephemeris_source(bh.FrameEphemerisSource.ANALYTIC)
try:
    bh.register_object("SC", lambda epoch: x_sc, bh.CelestialFrame.GCRF)
    r_graph = bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.NSW("SC"), epc
    )
    print(
        "Frame graph matches rotation_eci_to_nsw:",
        np.allclose(r_graph, bh.rotation_eci_to_nsw(x_sc, x_sun), atol=1e-9),
    )
    bh.clear_object_registry()
finally:
    bh.set_frame_ephemeris_source(bh.FrameEphemerisSource.AUTO)
