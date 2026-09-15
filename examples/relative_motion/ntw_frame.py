# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build the NTW frame on circular and eccentric orbits, compare it with RTN, and evaluate its rate about Earth and Mars.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

# Circular orbit: NTW and RTN coincide
x_circ = bh.state_koe_to_eci(
    np.array([bh.R_EARTH + 700e3, 0.0, 97.8, 15.0, 30.0, 45.0]), bh.AngleFormat.DEGREES
)
print(
    "Circular: NTW equals RTN:",
    np.allclose(bh.rotation_ntw_to_eci(x_circ), bh.rotation_rtn_to_eci(x_circ)),
)

# Eccentric orbit: the T axis leads R by the flight-path angle
x_ecc = bh.state_koe_to_eci(
    np.array([bh.R_EARTH + 700e3, 0.1, 97.8, 15.0, 30.0, 45.0]), bh.AngleFormat.DEGREES
)
r_ntw = bh.rotation_ntw_to_eci(x_ecc)
r_rtn = bh.rotation_rtn_to_eci(x_ecc)
gamma = np.degrees(np.arcsin(np.dot(r_ntw[:, 1], r_rtn[:, 0])))
print(f"Eccentric: flight-path angle between T and RTN T axis: {gamma:.3f} deg")

# Rates: the velocity direction turns slower than the position direction near periapsis
print(
    f"omega_ntw z: {bh.omega_ntw(x_ecc)[2]:.6e} rad/s, omega_rtn z: {bh.omega_rtn(x_ecc)[2]:.6e} rad/s"
)

# About Mars, pass the gravitational parameter explicitly
x_mars = bh.state_koe_to_eci(
    np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0]), bh.AngleFormat.DEGREES
)
print(
    f"omega_ntw_for_body about Mars: {bh.omega_ntw_for_body(x_mars, bh.GM_MARS)[2]:.6e} rad/s"
)

# Relative state round trip
x_rel = np.array([200.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
x_deputy = bh.state_ntw_to_eci(x_ecc, x_rel)
print(
    f"Round trip error: {np.linalg.norm(bh.state_eci_to_ntw(x_ecc, x_deputy) - x_rel):.2e}"
)

# Frame graph
epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
bh.register_object("SC", lambda epoch: x_ecc, bh.CelestialFrame.GCRF)
r_graph = bh.rotation_frame_to_frame(
    bh.CelestialFrame.GCRF, bh.ReferenceFrame.NTW("SC"), epc
)
print(
    "Frame graph matches rotation_eci_to_ntw:",
    np.allclose(r_graph, bh.rotation_eci_to_ntw(x_ecc)),
)
bh.clear_object_registry()
