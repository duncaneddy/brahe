# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
Download the FK5 catalog, look up a star by its running number, and filter
it by magnitude and cone search.
"""

import brahe as bh

# Download the FK5 catalog (cached permanently after the first download)
fk5 = bh.datasets.star_catalogs.get_fk5()
print(f"Loaded {len(fk5)} FK5 records")

# Look up a specific star by its FK5 running number
rec = fk5.get_by_id(699)
vmag_str = f"{rec.vmag:.2f}" if rec.vmag is not None else "N/A"
print(
    f"FK5 699: {rec.name() or rec.id()}, ra={rec.ra:.4f} deg, dec={rec.dec:.4f} deg, vmag={vmag_str}"
)

# Magnitude filter: keeps vmag <= max_mag (smaller/more negative is brighter)
bright = fk5.filter_by_magnitude(3.0)
print(f"Stars brighter than Vmag 3.0: {len(bright)}")

# Cone search around a right ascension/declination, in degrees
nearby = fk5.filter_by_cone(101.28, -16.72, 5.0, bh.AngleFormat.DEGREES)
print(f"Stars within 5 deg of (101.28, -16.72): {len(nearby)}")

# Chained: bright stars within 5 degrees of a target (filter methods return a
# new catalog instance, so the original catalog is never modified)
bright_nearby = fk5.filter_by_magnitude(3.0).filter_by_cone(
    101.28, -16.72, 5.0, bh.AngleFormat.DEGREES
)
print(f"Bright stars within 5 deg of target: {len(bright_nearby)}")
for r in bright_nearby.records():
    print(f"  {r.name() or r.id()}: Vmag={r.vmag:.2f}")
