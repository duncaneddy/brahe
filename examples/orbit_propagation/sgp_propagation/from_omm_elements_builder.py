# /// script
# dependencies = ["brahe"]
# ///
"""
Constructing an SGPPropagator from OMM elements using the builder API.

The builder takes the eight required OMM inputs -- epoch, mean_motion,
eccentricity, inclination, raan, arg_of_pericenter, mean_anomaly, and
norad_id -- directly as arguments to builder(). Optional inputs such as
object_name and bstar default when omitted and are set through chained
setters.
"""

import brahe as bh

bh.initialize_eop()

# ISS OMM mean elements
prop = (
    bh.SGPPropagator.builder(
        bh.Epoch.from_datetime(2025, 11, 29, 20, 1, 44.058144, 0.0, bh.TimeSystem.UTC),
        15.49193835,  # mean_motion (rev/day)
        0.0003723,  # eccentricity
        51.6312,  # inclination (degrees)
        206.3646,  # raan (degrees)
        184.1118,  # arg_of_pericenter (degrees)
        175.9840,  # mean_anomaly (degrees)
        25544,  # norad_id
    )
    .object_name("ISS (ZARYA)")
    .bstar(0.15237e-3)
    .build()
)

state = prop.state(prop.epoch)
position_magnitude = (state[0] ** 2 + state[1] ** 2 + state[2] ** 2) ** 0.5

print(f"NORAD ID: {prop.norad_id}")
print(f"Position magnitude: {position_magnitude / 1e3:.1f} km")
