# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
This example demonstrates how to calculate the orbital period of a satellite
given its semi-major axis using the Brahe library.
"""

import brahe as bh

bh.initialize_eop()

# Fetch the ISS from CelesTrak and create a propagator
client = bh.celestrak.CelestrakClient()
iss = client.get_sgp_propagator(catnr=25544, step_size=60.0)

# Compute upcoming passes of the ISS over San Francisco
passes = bh.location_accesses(
    bh.PointLocation(-122.4194, 37.7749, 0.0),  # San Francisco
    iss,
    bh.Epoch.now(),
    bh.Epoch.now() + 24 * 3600.0,  # Next 24 hours
    bh.ElevationConstraint(min_elevation_deg=10.0),
)
print(f"Number of passes in next 24 hours: {len(passes)}")
