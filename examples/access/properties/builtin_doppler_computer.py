# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using the built-in DopplerComputer for communication link analysis
"""

import brahe as bh

bh.initialize_eop()

# S-band downlink only (8.4 GHz)
doppler = bh.DopplerComputer(
    uplink_frequency=None,
    downlink_frequency=8.4e9,
    sampling_config=bh.SamplingConfig.fixed_interval(0.1, 0.0),  # 0.1 seconds
)
print(f"Downlink only: {doppler}")
# Downlink only: DopplerComputer(uplink=None, downlink=8.4e9 Hz, ...)

# Both uplink (2.0 GHz) and downlink (8.4 GHz)
doppler = bh.DopplerComputer(
    uplink_frequency=2.0e9,
    downlink_frequency=8.4e9,
    sampling_config=bh.SamplingConfig.fixed_count(100),
)
print(f"Both frequencies: {doppler}")
# Both frequencies: DopplerComputer(uplink=2.0e9 Hz, downlink=8.4e9 Hz, ...)

# Create a simple scenario to demonstrate usage
# ISS orbit
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2, 60.0).with_name("ISS")

epoch_start = propagator.epoch
epoch_end = epoch_start + 24 * 3600.0  # 24 hours

# Ground station (lon, lat, alt)
location = bh.PointLocation(-74.0060, 40.7128, 0.0)

# Compute accesses with Doppler
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
windows = bh.location_accesses(
    location,
    propagator,
    epoch_start,
    epoch_end,
    constraint,
    property_computers=[doppler],
)

# Access computed properties
window = windows[0]
doppler_data = window.properties.additional["doppler_downlink"]
times = doppler_data["times"]  # Seconds from window start
values = doppler_data["values"]  # Hz
print(
    f"\nFirst pass downlink Doppler shift range: {min(values):.1f} to {max(values):.1f} Hz"
)
# First pass Doppler shift range: -189220.9 to 189239.8 Hz
