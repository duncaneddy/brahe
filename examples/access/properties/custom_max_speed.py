# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Custom property computer that computes maximum ground speed during access
"""

import brahe as bh
import numpy as np

bh.initialize_eop()


class MaxSpeedComputer(bh.AccessPropertyComputer):
    """Computes maximum ground speed during access."""

    def sampling_config(self):
        # Sample every 0.5 seconds
        return bh.SamplingConfig.fixed_interval(0.5, 0.0)

    def compute(
        self, window, sample_times, sample_states_ecef, location_ecef, location_geodetic
    ):
        # Extract velocities from states
        velocities = sample_states_ecef[:, 3:6]
        speeds = np.linalg.norm(velocities, axis=1)
        max_speed = np.max(speeds)

        # Single value -> returns as scalar
        return {
            "max_ground_speed": max_speed,  # Will be stored as Scalar
        }

    def property_names(self):
        return ["max_ground_speed"]


# ISS orbit
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2, 60.0).with_name("ISS")

epoch_start = propagator.epoch
epoch_end = epoch_start + 24 * 3600.0  # 24 hours

# Ground station
location = bh.PointLocation(-74.0060, 40.7128, 0.0)

# Compute with custom property
max_speed = MaxSpeedComputer()
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
windows = bh.location_accesses(
    location,
    propagator,
    epoch_start,
    epoch_end,
    constraint,
    property_computers=[max_speed],
)

for window in windows:
    speed = window.properties.additional["max_ground_speed"]
    print(f"Max speed: {speed:.1f} m/s")

# Output example:
# Max speed: 7360.1 m/s
# Max speed: 7365.5 m/s
# Max speed: 7361.2 m/s
# Max speed: 7357.5 m/s
# Max speed: 7357.8 m/s
# Max speed: 7360.0 m/s
