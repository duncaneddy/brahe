# /// script
# dependencies = ["brahe"]
# ///
"""
Configure SGPPropagator output format (frame and representation)
"""

import brahe as bh

bh.initialize_eop()

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Create with ECEF Cartesian output
prop_ecef = bh.SGPPropagator.from_tle(line1, line2, 60.0)
prop_ecef.set_output_format(bh.OrbitFrame.ECEF, bh.OrbitRepresentation.CARTESIAN, None)

# Or with Keplerian output (ECI only)
prop_kep = bh.SGPPropagator.from_tle(line1, line2, 60.0)
prop_kep.set_output_format(
    bh.OrbitFrame.ECI, bh.OrbitRepresentation.KEPLERIAN, bh.AngleFormat.DEGREES
)

# Propagate to 1 hour after epoch
dt = 3600.0
prop_ecef.propagate_to(prop_ecef.epoch + dt)
prop_kep.propagate_to(prop_kep.epoch + dt)
print(f"ECEF position (km): {prop_ecef.current_state()[:3] / 1e3}")
state_kep = prop_kep.current_state()
print(
    f"Keplerian elements: [{state_kep[0]:.1f} km, {state_kep[1]:.4f}, {state_kep[2]:.4f}, "
    f"{state_kep[3]:.4f} deg, {state_kep[4]:.4f} deg, {state_kep[5]:.4f} deg]"
)

# Expected output:
# ECEF position (km): [ 5548.63233725  2869.31027561 -2526.64252368]
# Keplerian elements: [8198150.8 km, 0.1789, 47.9402, 249.8056 deg, 323.0545 deg, 4.5675 deg]
