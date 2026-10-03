# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build the EQW frame on an inclined orbit, check it against the ascending node and the argument of latitude, verify the equatorial-orbit fallback, and evaluate it through the frame graph.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

oe = np.array([bh.R_EARTH + 700e3, 0.1, 97.8, 15.0, 30.0, 45.0])
x_ecc = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
e_eqw = bh.rotation_eqw_to_eci(x_ecc)[:, 0]
raan = np.radians(15.0)
print(
    "E equals [cos Omega, sin Omega, 0]:",
    np.allclose(e_eqw, np.array([np.cos(raan), np.sin(raan), 0.0]), atol=1e-6),
)

r_eqw = bh.rotation_eci_to_eqw(x_ecc) @ x_ecc[:3]
r = np.linalg.norm(x_ecc[:3])
f = np.radians(bh.anomaly_mean_to_true(45.0, 0.1, angle_format=bh.AngleFormat.DEGREES))
u = np.radians(30.0) + f
print(
    "Own position in EQW equals r [cos u, sin u, 0]:",
    np.allclose(r_eqw, r * np.array([np.cos(u), np.sin(u), 0.0]), atol=1e-6),
)

# Equatorial orbit: the node is undefined, E falls back to the inertial x axis
oe_equatorial = np.array([bh.R_EARTH + 700e3, 0.1, 0.0, 15.0, 30.0, 45.0])
x_equatorial = bh.state_koe_to_eci(oe_equatorial, bh.AngleFormat.DEGREES)
e_equatorial = bh.rotation_eqw_to_eci(x_equatorial)[:, 0]
print(
    "Equatorial fallback: E equals [1, 0, 0]:",
    np.allclose(e_equatorial, np.array([1.0, 0.0, 0.0]), atol=1e-6),
)

# Frame graph
epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
bh.register_object("SC", lambda epoch: x_ecc, bh.CelestialFrame.GCRF)
r_graph = bh.rotation_frame_to_frame(
    bh.CelestialFrame.GCRF, bh.ReferenceFrame.EQW("SC"), epc
)
print(
    "Frame graph matches rotation_eci_to_eqw:",
    np.allclose(r_graph, bh.rotation_eci_to_eqw(x_ecc)),
)
bh.clear_object_registry()
