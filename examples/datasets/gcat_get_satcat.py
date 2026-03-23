# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Download and inspect the GCAT SATCAT catalog.

This example demonstrates downloading the SATCAT catalog and looking up
individual records by SATCAT number and JCAT identifier.
"""

import brahe as bh

# Download the SATCAT catalog (cached for 24 hours by default)
satcat = bh.datasets.gcat.get_satcat()
print(f"Loaded {len(satcat)} SATCAT records")

# Look up the ISS by NORAD SATCAT number
iss = satcat.get_by_satcat("25544")
if iss:
    print("\nISS (by SATCAT number 25544):")
    print(f"  JCAT:    {iss.jcat}")
    print(f"  Name:    {iss.name}")
    print(f"  Status:  {iss.status}")
    print(f"  Perigee: {iss.perigee} km")
    print(f"  Apogee:  {iss.apogee} km")
    print(f"  Inc:     {iss.inc}°")

# Look up by JCAT identifier
record = satcat.get_by_jcat("S049652")
if record:
    print(f"\nRecord by JCAT S049652: {record.name}")
