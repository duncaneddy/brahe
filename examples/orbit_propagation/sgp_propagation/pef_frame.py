# /// script
# dependencies = ["brahe"]
# ///
"""
Access satellite state in PEF (Pseudo-Earth-Fixed) frame
"""

import brahe as bh

bh.initialize_eop()

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

# Get state in PEF frame (TEME rotated by GMST)
state_pef = prop.state_pef(prop.epoch)
print(f"PEF position: {state_pef[:3] / 1e3}")
# Expected output:
# PEF position: [-3953.20574821  1427.51460044  5243.61453697]
