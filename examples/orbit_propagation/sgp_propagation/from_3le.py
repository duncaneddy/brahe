# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize SGPPropagator from 3-line TLE with satellite name
"""

import brahe as bh

bh.initialize_eop()

# 3-line TLE with satellite name
name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Create propagator with satellite name
prop = bh.SGPPropagator.from_3le(name, line1, line2, 60.0)

print(f"Satellite name: {prop.satellite_name}")
print(f"NORAD ID: {prop.norad_id}")
# Expected output:
# Satellite name: ISS (ZARYA)
# NORAD ID: 25544
