# /// script
# dependencies = ["brahe"]
# ///
"""
Query satellite states for multiple epochs at once
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

# Generate states for multiple orbits
orbital_period = 5400.0  # Approximate ISS period (seconds)
query_epochs = [prop.epoch + i * orbital_period for i in range(5)]
states_eci = prop.states_eci(query_epochs)

print(f"Generated {len(states_eci)} states over {len(query_epochs)} orbits")
for i, state in enumerate(states_eci):
    altitude = (np.linalg.norm(state[:3]) - bh.R_EARTH) / 1e3
    print(f"  Orbit {i}: altitude = {altitude:.1f} km")
# Expected output:
# Generated 5 states over 5 orbits
#   Orbit 0: altitude = 342.1 km
#   Orbit 1: altitude = 342.3 km
#   Orbit 2: altitude = 342.7 km
#   Orbit 3: altitude = 343.3 km
#   Orbit 4: altitude = 344.0 km
