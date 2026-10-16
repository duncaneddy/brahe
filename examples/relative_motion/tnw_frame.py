# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build the TNW frame on a circular orbit, compare it with RTN and NTW, and evaluate its rate about Earth and Mars.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

# Circular orbit: X = RTN's T, Z = RTN's N, and the in-plane axes are the RTN pair swapped and negated
x_circ = bh.state_koe_to_eci(
    np.array([bh.R_EARTH + 700e3, 0.0, 97.8, 15.0, 30.0, 45.0]), bh.AngleFormat.DEGREES
)
r_tnw = bh.rotation_tnw_to_eci(x_circ)
r_rtn = bh.rotation_rtn_to_eci(x_circ)
print(
    "Circular: TNW Y equals -RTN X, TNW X equals RTN Y, TNW Z equals RTN Z:",
    np.allclose(r_tnw[:, 1], -r_rtn[:, 0])
    and np.allclose(r_tnw[:, 0], r_rtn[:, 1])
    and np.allclose(r_tnw[:, 2], r_rtn[:, 2]),
)

# TNW is NTW with its in-plane axes reordered
r_ntw = bh.rotation_ntw_to_eci(x_circ)
print(
    "TNW is NTW reordered: X_TNW equals Y_NTW, Y_TNW equals -X_NTW:",
    np.allclose(r_tnw[:, 0], r_ntw[:, 1]) and np.allclose(r_tnw[:, 1], -r_ntw[:, 0]),
)

# Rates: TNW shares its rate with NTW
print(
    f"omega_tnw z: {bh.omega_tnw(x_circ)[2]:.6e} rad/s, omega_ntw z: {bh.omega_ntw(x_circ)[2]:.6e} rad/s"
)

# About Mars, pass the gravitational parameter explicitly
oe_mars = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
x_mars = bh.state_koe_to_inertial_for_body(
    oe_mars, bh.CentralBody.Mars, bh.AngleFormat.DEGREES
)
print(
    f"omega_tnw_for_body about Mars: {bh.omega_tnw_for_body(x_mars, bh.GM_MARS)[2]:.6e} rad/s"
)

# Relative state round trip: 1 km along-track, 200 m above along -N
x_rel = np.array([1000.0, -200.0, 0.0, 0.0, 0.0, 0.0])
x_deputy = bh.state_tnw_to_eci(x_circ, x_rel)
print(
    f"Round trip error: {np.linalg.norm(bh.state_eci_to_tnw(x_circ, x_deputy) - x_rel):.2e}"
)

# Frame graph
epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
bh.register_object("SC", lambda epoch: x_circ, bh.CelestialFrame.GCRF)
r_graph = bh.rotation_frame_to_frame(
    bh.CelestialFrame.GCRF, bh.ReferenceFrame.TNW("SC"), epc
)
print(
    "Frame graph matches rotation_eci_to_tnw:",
    np.allclose(r_graph, bh.rotation_eci_to_tnw(x_circ)),
)
bh.clear_object_registry()
