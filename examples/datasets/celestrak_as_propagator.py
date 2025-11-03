# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Convert CelesTrak TLE directly to SGP propagator.

This example shows how to get a satellite and convert it to a propagator
in a single step, which is the most common use case.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Get ISS as a propagator with 60-second step size
# The group hint ("stations") uses cached data for efficiency
iss_prop = bh.datasets.celestrak.get_tle_by_id_as_propagator(25544, 60.0, "stations")

print(f"Created propagator: {iss_prop.name}")
print(f"Epoch: {iss_prop.epoch}")

# Propagate forward 1 orbit period (~93 minutes for ISS)
iss_prop.propagate_to(iss_prop.epoch + 93.0 * 60.0)
state = iss_prop.current_state()

print("\nState after 1 orbit:")
print(f"  Position: [{state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f}] m")
print(f"  Velocity: [{state[3]:.1f}, {state[4]:.1f}, {state[5]:.1f}] m/s")

# Expected output:
# Created propagator: ISS (ZARYA)
# Epoch: 2025-11-02 10:09:34.283 UTC

# State after 1 orbit:
#   Position: [6451630.2, -2126316.1, 34427.2] m
#   Velocity: [2019.6, 5281.4, 6006.2] m/s
