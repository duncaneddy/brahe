# /// script
# dependencies = ["brahe", "numpy"]
# FLAGS = ["IGNORE"]
# ///
"""
Parallel propagation of multiple satellites

This example demonstrates using par_propagate_to() to efficiently propagate
multiple satellites to a target epoch in parallel, useful for constellation
analysis and Monte Carlo simulations.
"""

import brahe as bh
import numpy as np
import time

bh.initialize_eop()

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Create propagators of different types
propagators = []

# Create Keplerian propagator
oe = np.array([bh.R_EARTH + 500e3, 0.001, 98.0, 36.0, 10.0, 36.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
propagators.append(prop)

# Add SGP propagator
line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)
propagators.append(prop)


# Propagate all satellites in parallel
start = time.time()
bh.par_propagate_to(propagators, epoch + 86400.0)
parallel_time = time.time() - start

# STDERR:
# Traceback (most recent call last):
#   File ".../brahe/examples/orbit_propagation/functions/mixed_parallel_propagation.py", line 43, in <module>
#     bh.par_propagate_to(propagators, epoch + 86400.0)
#     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# TypeError: 'SGPPropagator' object cannot be converted to 'KeplerianPropagator'
