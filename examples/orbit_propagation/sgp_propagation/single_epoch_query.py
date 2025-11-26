# /// script
# dependencies = ["brahe"]
# ///
"""
Query satellite state at arbitrary epochs without stepping
"""

import brahe as bh

bh.initialize_eop()

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

# Query state 1 orbit later (doesn't add to trajectory)
query_epoch = prop.epoch + 5400.0  # ~90 minutes

state_eci = prop.state_eci(query_epoch)  # ECI Cartesian
state_ecef = prop.state_ecef(query_epoch)  # ECEF Cartesian
state_kep = prop.state_koe(query_epoch, bh.AngleFormat.DEGREES)  # Osculating Keplerian

print(
    f"ECI position: [{state_eci[0] / 1e3:.1f}, {state_eci[1] / 1e3:.1f}, "
    f"{state_eci[2] / 1e3:.1f}] km"
)
print(f"Osculating semi-major axis: {state_kep[0] / 1e3:.1f} km")

# Expected output:
# ECI position: [3822.2, -1684.2, 5264.9] km
# Osculating semi-major axis: 6725.4 km
