# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
Loading a CENTER_NAME/REF_FRAME pair that names a non-Earth origin.

OEMExample4.txt declares CENTER_NAME = MARS with REF_FRAME = EME2000: EME2000
orientation about Mars rather than Earth. OEM.to_trajectories resolves that
pair to CelestialFrame.Centered(MARS, EME2000) instead of the Earth-centered
EME2000 shorthand. Converting a sample to MCI only rotates (no ephemeris
needed, since both frames share Mars as their center); converting to GCRF
also translates by the Earth-Mars vector, which requires the DE440s planetary
ephemeris loaded below.
"""

import brahe as bh
from brahe.ccsds import OEM

bh.initialize_eop()
bh.load_common_spice_kernels()

oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
seg = oem.segments[0]
print(f"REF_FRAME = {seg.ref_frame}, CENTER_NAME = {seg.center_name}")

traj = oem.to_trajectories()[0]
print(f"\nTrajectory frame: {traj.frame}")

# The trajectory's frame is a CelestialFrame naming the (center, axes)
# pair; celestial_frame unwraps the ReferenceFrame and axes and center split
# the result back into its two halves.
traj_frame = traj.frame.celestial_frame
print(f"  Axes:   {traj_frame.axes}")
print(f"  Center: {traj_frame.center.name}")

# Same-center conversion: EME2000 to MCI is a rotation only, both centered on
# Mars.
traj_mci = traj.to_frame(bh.CelestialFrame.MCI)
epc, x_mci = traj_mci.get(0)
print(f"\nState at {epc} in MCI (Mars-centered):")
print(
    f"  Position (km): [{x_mci[0] / 1e3:.3f}, {x_mci[1] / 1e3:.3f}, {x_mci[2] / 1e3:.3f}]"
)
print(f"  Velocity (m/s): [{x_mci[3]:.3f}, {x_mci[4]:.3f}, {x_mci[5]:.3f}]")

# Cross-center conversion: EME2000/Mars to GCRF/Earth adds the Earth-Mars
# translation from the loaded SPK kernels.
traj_gcrf = traj.to_frame(bh.CelestialFrame.GCRF)
_, x_gcrf = traj_gcrf.get(0)
print(f"\nState at {epc} in GCRF (Earth-centered):")
print(
    f"  Position (km): [{x_gcrf[0] / 1e3:.3e}, {x_gcrf[1] / 1e3:.3e}, {x_gcrf[2] / 1e3:.3e}]"
)
print(f"  Velocity (m/s): [{x_gcrf[3]:.3f}, {x_gcrf[4]:.3f}, {x_gcrf[5]:.3f}]")

assert traj_frame.center == bh.NAIFId.MARS
assert traj_frame.axes == bh.FrameAxes.EME2000
print("\nExample validated successfully!")
