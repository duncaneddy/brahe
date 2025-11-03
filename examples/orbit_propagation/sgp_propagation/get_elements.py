# /// script
# dependencies = ["brahe"]
# ///
"""
Extract Keplerian orbital elements from TLE data
"""

import brahe as bh

bh.initialize_eop()

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

# Extract Keplerian elements from TLE
elements_deg = prop.get_elements(bh.AngleFormat.DEGREES)
elements_rad = prop.get_elements(bh.AngleFormat.RADIANS)

print(f"Semi-major axis: {elements_deg[0] / 1e3:.1f} km")
print(f"Eccentricity: {elements_deg[1]:.6f}")
print(f"Inclination: {elements_deg[2]:.4f} degrees")
print(f"RAAN: {elements_deg[3]:.4f} degrees")
print(f"Argument of perigee: {elements_deg[4]:.4f} degrees")
print(f"Mean anomaly: {elements_deg[5]:.4f} degrees")
# Expected output:
# Semi-major axis: 6758.7 km
# Eccentricity: 0.000670
# Inclination: 51.6416 degrees
# RAAN: 247.4627 degrees
# Argument of perigee: 130.5360 degrees
# Mean anomaly: 325.0288 degrees
