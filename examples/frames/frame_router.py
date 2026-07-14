# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Route an orbital state between frames with different centers via the frame router.
"""

import brahe as bh
import numpy as np

# Initialize EOP and the DE440s ephemeris used to re-center GCRF (Earth) to
# LCI (Moon) inside the router.
bh.initialize_eop()
bh.load_common_spice_kernels()

epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# A GCRF (Earth-centered inertial) state, built from Keplerian elements.
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 15.0, 30.0, 45.0])
x_gcrf = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# state_frame_to_frame routes through ICRF and re-centers Earth -> Moon, so the
# same physical state is now expressed relative to the Moon.
x_lci = bh.state_frame_to_frame(
    bh.ReferenceFrame.GCRF, bh.ReferenceFrame.LCI, epc, x_gcrf
)

# rotation_frame_to_frame returns only the 3x3 axis rotation (no re-centering);
# GCRF -> MCMF uses the compiled-in WGCCRE Mars model and needs no kernel.
r_gcrf_to_mcmf = bh.rotation_frame_to_frame(
    bh.ReferenceFrame.GCRF, bh.ReferenceFrame.MCMF, epc
)

print(f"Epoch: {epc}")
print("\nGCRF state (Earth-centered inertial):")
print(
    f"  Position (km): [{x_gcrf[0] / 1e3:.3f}, {x_gcrf[1] / 1e3:.3f}, {x_gcrf[2] / 1e3:.3f}]"
)
print("\nSame state expressed in LCI (Moon-centered inertial):")
print(
    f"  Position (km): [{x_lci[0] / 1e3:.3f}, {x_lci[1] / 1e3:.3f}, {x_lci[2] / 1e3:.3f}]"
)

# The GCRF->LCI position shift equals the Moon's distance from Earth (~384,400 km),
# since both frames share ICRF axes and differ only in origin.
offset = np.linalg.norm(x_lci[:3] - x_gcrf[:3])
print(f"\nGCRF->LCI position offset (km): {offset / 1e3:.1f}")
assert 350_000e3 < offset < 410_000e3

# The returned rotation is a proper orthonormal direction cosine matrix.
assert np.allclose(r_gcrf_to_mcmf @ r_gcrf_to_mcmf.T, np.eye(3), atol=1e-9)
print("\nExample validated successfully!")
