# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build the SEZ frame of a ground station, find a satellite's azimuth and elevation, compare the rate of a static and a moving site, and evaluate it through the frame graph.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

r_gs = bh.position_geodetic_to_ecef(
    np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES
)
x_gs = np.concatenate([r_gs, np.zeros(3)])

oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
x_sat = bh.state_gcrf_to_itrf(epc, x_eci)

x_rel_sez = bh.state_ecef_to_sez(x_gs, x_sat)
azel = bh.position_sez_to_azel(x_rel_sez[:3], bh.AngleFormat.DEGREES)
print(f"Azimuth: {azel[0]:.3f} deg, Elevation: {azel[1]:.3f} deg")

omega_static = bh.omega_sez(x_gs)
print("Static station omega norm is zero:", np.linalg.norm(omega_static) == 0.0)

x_aircraft = np.concatenate([r_gs, np.array([-120.0, 180.0, 90.0])])
omega_moving = bh.omega_sez(x_aircraft)
print(f"Moving site omega norm: {np.linalg.norm(omega_moving):.3e} rad/s")

bh.register_object("GS", lambda epoch: x_gs, bh.CelestialFrame.ITRF)
rel_graph = bh.state_frame_to_frame(
    bh.CelestialFrame.ITRF, bh.ReferenceFrame.SEZ("GS"), epc, x_sat
)
print(
    "Frame graph matches state_ecef_to_sez:",
    np.linalg.norm(rel_graph - x_rel_sez) < 1e-6,
)
bh.clear_object_registry()
