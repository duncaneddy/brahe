# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Combining built-in and custom property computers
"""

import brahe as bh
import numpy as np

bh.initialize_eop()


class MaxSpeedComputer(bh.AccessPropertyComputer):
    """Computes maximum ground speed during access."""

    def sampling_config(self):
        return bh.SamplingConfig.fixed_interval(0.5, 0.0)

    def compute(
        self, window, sample_times, sample_states_ecef, location_ecef, location_geodetic
    ):
        velocities = sample_states_ecef[:, 3:6]
        speeds = np.linalg.norm(velocities, axis=1)
        max_speed = np.max(speeds)
        return {"max_ground_speed": max_speed}

    def property_names(self):
        return ["max_ground_speed"]


# Mix built-in and custom computers
doppler = bh.DopplerComputer(
    uplink_frequency=None,
    downlink_frequency=2.2e9,
    sampling_config=bh.SamplingConfig.fixed_interval(0.1, 0.0),
)

range_comp = bh.RangeComputer(sampling_config=bh.SamplingConfig.midpoint())

custom_comp = MaxSpeedComputer()

# Setup scenario
# ISS orbit
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2, 60.0).with_name("ISS")

epoch_start = propagator.epoch
epoch_end = epoch_start + 24 * 3600.0  # 24 hours

# Ground station
location = bh.PointLocation(-74.0060, 40.7128, 0.0)

# Compute with all property computers
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
windows = bh.location_accesses(
    location,
    propagator,
    epoch_start,
    epoch_end,
    constraint,
    property_computers=[doppler, range_comp, custom_comp],
)

# All properties available in results
window = windows[0]
props = window.properties.additional

doppler_data = props["doppler_downlink"]
range_data = props["range"]
max_speed = props["max_ground_speed"]

print(f"Doppler: {len(doppler_data['values'])} samples")
print(f"Range: {range_data / 1000:.1f} km")
print(f"Max speed: {max_speed:.1f} m/s")
# Doppler: 3777 samples
# Range: 658.4 km
# Max speed: 7360.1 m/s
