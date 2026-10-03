# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build the ENZ frame of a ground station, find a satellite's azimuth and elevation, show it as the SEZ permutation, compare the rate of a moving site to the permuted SEZ rate, and evaluate it through the frame graph.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

r_gs = bh.position_geodetic_to_ecef(
    np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES
)
x_gs = np.concatenate([r_gs, np.zeros(3)])

oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 105.0])
x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
x_sat = bh.state_gcrf_to_itrf(epc, x_eci)

x_rel_enz = bh.state_ecef_to_enz(x_gs, x_sat)
azel = bh.position_enz_to_azel(x_rel_enz[:3], bh.AngleFormat.DEGREES)
print(f"Azimuth: {azel[0]:.3f} deg, Elevation: {azel[1]:.3f} deg")

x_rel_sez = bh.state_ecef_to_sez(x_gs, x_sat)
is_sez_permutation = (
    x_rel_enz[0] == x_rel_sez[1]
    and x_rel_enz[1] == -x_rel_sez[0]
    and x_rel_enz[2] == x_rel_sez[2]
)
print("ENZ columns are the SEZ permutation (E=E, N=-S, Z=Z):", is_sez_permutation)

x_aircraft = np.concatenate([r_gs, np.array([-120.0, 180.0, 90.0])])
omega_enz = bh.omega_enz(x_aircraft)
omega_sez = bh.omega_sez(x_aircraft)
omega_permuted = np.array([omega_sez[1], -omega_sez[0], omega_sez[2]])
print(
    "Moving site omega_enz matches the permuted SEZ rate:",
    np.linalg.norm(omega_enz - omega_permuted) < 1e-15,
)

bh.register_object("GS", lambda epoch: x_gs, bh.CelestialFrame.ITRF)
rel_graph = bh.state_frame_to_frame(
    bh.CelestialFrame.ITRF, bh.ReferenceFrame.ENZ("GS"), epc, x_sat
)
print(
    "Frame graph matches state_ecef_to_enz:",
    np.linalg.norm(rel_graph - x_rel_enz) < 1e-6,
)
bh.clear_object_registry()
