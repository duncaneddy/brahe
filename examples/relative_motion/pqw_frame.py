# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build the PQW frame on an eccentric orbit, check it against the true anomaly and the circular-orbit node-line fallback, verify the Mars `_for_body` periapsis direction, and evaluate it through the frame graph.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

oe = np.array([bh.R_EARTH + 700e3, 0.1, 97.8, 15.0, 30.0, 45.0])
x_ecc = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
r_pqw = bh.rotation_eci_to_pqw(x_ecc) @ x_ecc[:3]
r = np.linalg.norm(x_ecc[:3])
f = np.radians(bh.anomaly_mean_to_true(45.0, 0.1, angle_format=bh.AngleFormat.DEGREES))
print(
    "Own position in PQW equals r [cos f, sin f, 0]:",
    np.allclose(r_pqw, r * np.array([np.cos(f), np.sin(f), 0.0]), atol=1e-6),
)

# Circular orbit: periapsis is undefined, P falls back to the ascending node
oe_circ = np.array([bh.R_EARTH + 700e3, 0.0, 97.8, 15.0, 30.0, 45.0])
x_circ = bh.state_koe_to_eci(oe_circ, bh.AngleFormat.DEGREES)
p_circ = bh.rotation_pqw_to_eci(x_circ)[:, 0]
raan = np.radians(15.0)
print(
    "Circular fallback: P equals [cos Omega, sin Omega, 0]:",
    np.allclose(p_circ, np.array([np.cos(raan), np.sin(raan), 0.0])),
)

# About Mars, the periapsis direction is the position at mean anomaly zero
oe_mars = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
x_mars = bh.state_koe_to_inertial_for_body(
    oe_mars, bh.CentralBody.Mars, bh.AngleFormat.DEGREES
)
oe_mars_periapsis = oe_mars.copy()
oe_mars_periapsis[5] = 0.0
x_mars_periapsis = bh.state_koe_to_inertial_for_body(
    oe_mars_periapsis, bh.CentralBody.Mars, bh.AngleFormat.DEGREES
)
p_mars = bh.rotation_pqw_to_inertial_for_body(x_mars, bh.GM_MARS)[:, 0]
r_periapsis_hat = x_mars_periapsis[:3] / np.linalg.norm(x_mars_periapsis[:3])
print(
    "Mars periapsis direction matches the position at M = 0:",
    np.allclose(p_mars, r_periapsis_hat),
)

# Frame graph
epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
bh.register_object("SC", lambda epoch: x_ecc, bh.CelestialFrame.GCRF)
r_graph = bh.rotation_frame_to_frame(
    bh.CelestialFrame.GCRF, bh.ReferenceFrame.PQW("SC"), epc
)
print(
    "Frame graph matches rotation_eci_to_pqw:",
    np.allclose(r_graph, bh.rotation_eci_to_pqw(x_ecc)),
)
bh.clear_object_registry()
