# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build the LVLH frame of a satellite, check it against RTN, and express a
deputy's relative state in it, both directly and through the frame graph.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

# Chief in a 700 km, e = 0.01, sun-synchronous orbit
oe_chief = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)

# LVLH axes (CCSDS/SANA): Z nadir, Y anti-normal, X = Y x Z
r_lvlh_to_eci = bh.rotation_lvlh_to_eci(x_chief)
r_rtn_to_eci = bh.rotation_rtn_to_eci(x_chief)
print("LVLH X equals RTN T:", np.allclose(r_lvlh_to_eci[:, 0], r_rtn_to_eci[:, 1]))
print("LVLH Z equals -RTN R:", np.allclose(r_lvlh_to_eci[:, 2], -r_rtn_to_eci[:, 0]))

# Frame rate: the orbit rate about the -Y axis
omega = bh.omega_lvlh(x_chief)
print(f"omega_lvlh (rad/s): [{omega[0]:.3e}, {omega[1]:.3e}, {omega[2]:.3e}]")

# Deputy 1 km ahead and 200 m above the chief in the rotating LVLH frame
x_rel = np.array([1000.0, 0.0, -200.0, 0.0, 0.0, 0.0])
x_deputy = bh.state_lvlh_to_eci(x_chief, x_rel)
x_rel_back = bh.state_eci_to_lvlh(x_chief, x_deputy)
print(f"Relative state round trip error: {np.linalg.norm(x_rel_back - x_rel):.2e}")

# Frame graph: the same rotation from a registered object
epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
bh.register_object("CHIEF", lambda epoch: x_chief, bh.CelestialFrame.GCRF)
r_graph = bh.rotation_frame_to_frame(
    bh.CelestialFrame.GCRF, bh.ReferenceFrame.LVLH("CHIEF"), epc
)
print(
    "Frame graph matches rotation_eci_to_lvlh:",
    np.allclose(r_graph, bh.rotation_eci_to_lvlh(x_chief)),
)
bh.clear_object_registry()
