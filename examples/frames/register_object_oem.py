# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Register an OEM ephemeris as an object with `OEM.register_for`, then query it
through the object's RTN orbit-relative frame.
"""

import numpy as np

import brahe as bh
from brahe.ccsds import OEM

bh.clear_object_registry()

# OEM.register_for is a one-liner: it converts the ephemeris segment to a
# trajectory, wraps it as a state provider, and registers it under a name.
oem = OEM.from_file("test_assets/ccsds/oem/OEMExample5.txt")
oem.register_for("ISS")
print(f"Registered objects: {bh.registered_objects()}")

# The registered object anchors ReferenceFrame.RTN("ISS"): its origin is the object's
# GCRF position, interpolated from the OEM ephemeris.
epc = oem.segments[0].start_time + 300.0
x_rtn_origin = bh.state_frame_to_frame(
    bh.ReferenceFrame.RTN("ISS"), bh.CelestialFrame.GCRF, epc, np.zeros(6)
)
print(f"\nISS position at {epc}: {x_rtn_origin[:3] / 1e3} km")

bh.clear_object_registry()
print("\nExample validated successfully!")
