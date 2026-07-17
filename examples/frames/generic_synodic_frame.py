# /// script
# dependencies = ["brahe", "numpy"]
# FLAGS = ["NETWORK"]
# ///
"""
Construct a generic two-body Synodic frame and transform a state into it.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()
bh.load_common_spice_kernels()

epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# A generic Synodic frame isn't limited to the named EMR/SER/GSE instances:
# any two SPK-covered bodies work as the primary/secondary pair. Here the
# Sun-Mars barycenter defines a Sun-Mars rotating frame.
frame = bh.ReferenceFrame.Synodic(bh.SynodicOrigin.Barycenter, 10, 4)

# A LEO state in GCRF, transformed into the Sun-Mars rotating frame through
# the frame router.
x_gcrf = bh.state_koe_to_eci(
    np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0]),
    bh.AngleFormat.DEGREES,
)
x_syn = bh.state_frame_to_frame(bh.ReferenceFrame.GCRF, frame, epc, x_gcrf)

print(f"Epoch: {epc}")
print(f"GCRF state (km): {np.array2string(x_gcrf[:3] / 1e3, precision=3)}")
print(f"Sun-Mars synodic state (km): {np.array2string(x_syn[:3] / 1e3, precision=3)}")

# The frame exposes its origin/primary/secondary back through properties.
assert frame.synodic_origin == bh.SynodicOrigin.Barycenter
assert frame.synodic_primary == 10
assert frame.synodic_secondary == 4

# Round-tripping back to GCRF recovers the original state.
x_back = bh.state_frame_to_frame(frame, bh.ReferenceFrame.GCRF, epc, x_syn)
np.testing.assert_allclose(x_back, x_gcrf, atol=1e-6)

print("\nExample validated successfully!")
