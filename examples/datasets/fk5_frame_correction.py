# /// script
# dependencies = ["brahe", "numpy"]
# FLAGS = ["NETWORK"]
# ///
"""
Transform an FK5 catalog direction into GCRF for the IAU 2006/2000A frames.

FK5 positions are referred to the mean equator and equinox of J2000.0 - the
same axes Brahe realizes as EME2000 - not the ICRS/GCRF axes used by the
IAU 2006/2000A (CIO-based) frame transformations. An FK5 direction must
therefore be rotated from EME2000 to GCRF (a small ~23 mas frame-bias
rotation) before it is mixed with a GCRF state or fed into the GCRF -> ITRF
transform. Hipparcos and Tycho-2 are already on the ICRS/GCRF axes and need
no such rotation - only proper-motion propagation.
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Download FK5 and look up a star by its running number
fk5 = bh.datasets.star_catalogs.get_fk5()
rec = fk5.get_by_id(699)
print(f"FK5 699: {rec.name() or rec.id()}")
print(
    f"  Catalog (EME2000) position, J2000.0: RA={rec.ra:.6f} deg, Dec={rec.dec:.6f} deg"
)

# Propagate the catalog position to the observation epoch with proper motion.
# The FK5 axes are fixed and do not depend on the position epoch, so the
# propagated (ra, dec) is still expressed on the EME2000 axes.
epc = bh.Epoch.from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, bh.UTC)
ra, dec = rec.radec_at_epoch(epc, bh.AngleFormat.DEGREES)
print(f"  Propagated to epoch:                 RA={ra:.6f} deg, Dec={dec:.6f} deg")

# Direction as a unit vector on the EME2000 axes ...
u_eme2000 = bh.position_radec_to_inertial(
    np.array([ra, dec, 1.0]), bh.AngleFormat.DEGREES
)

# ... rotated onto the GCRF/ICRS axes the IAU 2006/2000A transforms expect.
u_gcrf = bh.position_eme2000_to_gcrf(u_eme2000)

# The frame bias is a small (~23 mas) but non-zero rotation.
sep_rad = np.arctan2(
    np.linalg.norm(np.cross(u_eme2000, u_gcrf)), np.dot(u_eme2000, u_gcrf)
)
print(f"\nEME2000 -> GCRF frame-bias shift: {np.degrees(sep_rad) * 3.6e6:.2f} mas")

# The GCRF direction can now be rotated into the Earth-fixed (ITRF) frame with
# the IAU 2006/2000A CIO-based transform (e.g. for a topocentric pointing).
u_itrf = bh.rotation_eci_to_ecef(epc) @ u_gcrf
print(f"GCRF unit vector: [{u_gcrf[0]:.6f}, {u_gcrf[1]:.6f}, {u_gcrf[2]:.6f}]")
print(f"ITRF unit vector: [{u_itrf[0]:.6f}, {u_itrf[1]:.6f}, {u_itrf[2]:.6f}]")
