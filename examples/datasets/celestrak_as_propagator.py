# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Convert CelesTrak GP data directly to SGP propagator.

This example shows how to query a satellite from CelesTrak and convert it
to a propagator in a few steps, which is the most common use case.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Get an SGP4 propagator for the ISS directly from CelesTrak
client = bh.celestrak.CelestrakClient()
iss_prop = client.get_sgp_propagator(catnr=25544, step_size=60.0)

print(f"Created propagator: {iss_prop.get_name()}")
print(f"Epoch: {iss_prop.epoch}")

# Propagate forward 1 orbit period (~93 minutes for ISS)
iss_prop.propagate_to(iss_prop.epoch + bh.orbital_period(iss_prop.semi_major_axis))
state = iss_prop.current_state()

print("\nState after 1 orbit:")
print(f"  Position: [{state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f}] m")
print(f"  Velocity: [{state[3]:.1f}, {state[4]:.1f}, {state[5]:.1f}] m/s")
