# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Parallel propagation of mixed satellite types

This example demonstrates using par_propagate_to() with a list that mixes
propagator types. Propagators are grouped by type internally and each group is
propagated in parallel, which is useful for constellation analysis and Monte
Carlo simulations that combine analytic and SGP4 orbit models.
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Anchor every propagator to a common epoch so a single target propagates them
# all by the same time span. Here we use an ISS TLE and its epoch.
line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
sgp = bh.SGPPropagator.from_tle(line1, line2, 60.0)
epoch = sgp.epoch

# Build a Keplerian propagator at the same epoch.
oe = np.array([bh.R_EARTH + 500e3, 0.001, 98.0, 36.0, 10.0, 36.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
kep = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)

# A single list mixing Keplerian and SGP propagators.
propagators = [kep, sgp]

# Propagate every satellite forward one hour, in parallel, in place.
target = epoch + 3600.0
bh.par_propagate_to(propagators, target)

for prop in propagators:
    assert prop.current_epoch() == target

print(f"Propagated {len(propagators)} mixed-type propagators one hour to {target}.")
print(f"  Keplerian position (m): {kep.current_state()[:3]}")
print(f"  SGP position (m):       {sgp.current_state()[:3]}")
