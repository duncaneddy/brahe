# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize SGPPropagator from 2-line TLE data
"""

import brahe as bh
import numpy as np

bh.initialize_eop()  # Required for accurate frame transformations

# ISS TLE data (example)
line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Create propagator with 60-second step size
prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

print(f"NORAD ID: {prop.norad_id}")
print(f"TLE epoch: {prop.epoch}")
print(
    f"Initial position magnitude: {np.linalg.norm(prop.initial_state()[:3]) / 1e3:.1f} km"
)
# Expected output:
# NORAD ID: 25544
# TLE epoch: 2008-09-20 12:25:40.104 UTC
# Initial position magnitude: 6720.2 km
