# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build the VNC frame on a circular orbit, compare it with TNW and NTW, and evaluate its rate about Earth and Mars.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

x_circ = bh.state_koe_to_eci(
    np.array([bh.R_EARTH + 700e3, 0.0, 97.8, 15.0, 30.0, 45.0]), bh.AngleFormat.DEGREES
)
r_vnc = bh.rotation_vnc_to_eci(x_circ)
r_tnw = bh.rotation_tnw_to_eci(x_circ)
r_ntw = bh.rotation_ntw_to_eci(x_circ)
print(
    "VNC is TNW and NTW reordered: Y_VNC equals Z_TNW, Z_VNC equals X_NTW:",
    np.allclose(r_vnc[:, 1], r_tnw[:, 2]) and np.allclose(r_vnc[:, 2], r_ntw[:, 0]),
)

# Circular orbit: X = RTN's T, Y = RTN's N, Z = RTN's R
r_rtn = bh.rotation_rtn_to_eci(x_circ)
print(
    "Circular: VNC X equals RTN T, Y equals RTN N, Z equals RTN R:",
    np.allclose(r_vnc[:, 0], r_rtn[:, 1])
    and np.allclose(r_vnc[:, 1], r_rtn[:, 2])
    and np.allclose(r_vnc[:, 2], r_rtn[:, 0]),
)

# Rate: VNC turns about Y, the orbit normal
print(f"omega_vnc y: {bh.omega_vnc(x_circ)[1]:.6e} rad/s")

# About Mars, pass the gravitational parameter explicitly
oe_mars = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
x_mars = bh.state_koe_to_inertial_for_body(
    oe_mars, bh.CentralBody.Mars, bh.AngleFormat.DEGREES
)
print(
    f"omega_vnc_for_body about Mars: {bh.omega_vnc_for_body(x_mars, bh.GM_MARS)[1]:.6e} rad/s"
)

# Relative state round trip: 1 km along-track, 200 m outward along the co-normal
x_rel = np.array([1000.0, 0.0, 200.0, 0.0, 0.0, 0.0])
x_deputy = bh.state_vnc_to_eci(x_circ, x_rel)
print(
    f"Round trip error: {np.linalg.norm(bh.state_eci_to_vnc(x_circ, x_deputy) - x_rel):.2e}"
)

# Frame graph
epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
bh.register_object("SC", lambda epoch: x_circ, bh.CelestialFrame.GCRF)
r_graph = bh.rotation_frame_to_frame(
    bh.CelestialFrame.GCRF, bh.ReferenceFrame.VNC("SC"), epc
)
print(
    "Frame graph matches rotation_eci_to_vnc:",
    np.allclose(r_graph, bh.rotation_eci_to_vnc(x_circ)),
)
bh.clear_object_registry()
