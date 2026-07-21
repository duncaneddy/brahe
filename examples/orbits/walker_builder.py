# /// script
# dependencies = ["brahe"]
# ///
"""
Constructing a WalkerConstellationGenerator using the builder API.

The builder takes the six required inputs -- t, p, f, semi_major_axis,
inclination, and epoch -- directly as arguments to builder(). Optional
geometry fields (eccentricity, argument_of_perigee, reference_raan,
reference_mean_anomaly, pattern) default to the same values as the flat
constructor and are set through chained setters.
"""

import brahe as bh

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# GPS-like 24:6:2 Walker Delta constellation
walker = (
    bh.WalkerConstellationGenerator.builder(
        24,  # t
        6,  # p
        2,  # f
        bh.R_EARTH + 20200e3,  # semi_major_axis
        55.0,  # inclination (degrees)
        epoch,
    )
    .base_name("GPS")
    .build()
)

print(f"Total satellites: {walker.total_satellites}")
print(f"Number of planes: {walker.num_planes}")
print(f"Pattern: {walker.pattern}")

propagators = walker.as_keplerian_propagators(60.0)
print(f"Generated {len(propagators)} Keplerian propagators")
print(f"First propagator name: {propagators[0].get_name()}")
