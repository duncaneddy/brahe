# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
Filter the Hipparcos catalog by magnitude and cone search to locate a star,
then propagate its position to a different epoch with `radec_at_epoch`.
"""

import brahe as bh

hipparcos = bh.datasets.star_catalogs.get_hipparcos()

# Magnitude filter: keeps vmag <= max_mag (smaller/more negative is brighter)
bright = hipparcos.filter_by_magnitude(2.0)
print(f"Stars brighter than Vmag 2.0: {len(bright)}")

# Cone search around Sirius' catalog position, in degrees
nearby = bright.filter_by_cone(101.28, -16.72, 5.0, bh.AngleFormat.DEGREES)
print(f"Bright stars within 5 deg of Sirius: {len(nearby)}")
for r in nearby.records():
    print(f"  {r.name() or r.id()}: Vmag={r.vmag:.2f}")

# Propagate Sirius (HIP 32349) to a future epoch using radec_at_epoch (same
# proper-motion transformation as apply_proper_motion, applied directly to a
# catalog record)
sirius = hipparcos.get_by_id(32349)
epc = bh.Epoch.from_datetime(2030, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
ra, dec = sirius.radec_at_epoch(epc, bh.AngleFormat.DEGREES)
print(f"\nSirius at J2030.0:   RA={ra:.6f} deg, Dec={dec:.6f} deg")
print(f"Sirius at J1991.25:  RA={sirius.ra:.6f} deg, Dec={sirius.dec:.6f} deg")
