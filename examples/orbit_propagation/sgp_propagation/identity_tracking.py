# /// script
# dependencies = ["brahe"]
# ///
"""
Assign custom names and IDs to propagators
"""

import brahe as bh

bh.initialize_eop()

line0 = "ISS (ZARYA)"
line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Create propagator and set identity
prop = bh.SGPPropagator.from_3le(line0, line1, line2, 60.0)

print(f"Name: {prop.name}")
print(f"ID: {prop.id}")
print(f"NORAD ID from TLE: {prop.norad_id}")
# Expected output:
# Name: ISS (ZARYA)
# ID: 25544
# NORAD ID from TLE: 25544
