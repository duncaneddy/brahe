# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using the built-in RangeRateComputer to compute line-of-sight velocity
"""

import brahe as bh

bh.initialize_eop()

# Compute range rate every 0.5 seconds
range_rate = bh.RangeRateComputer(
    sampling_config=bh.SamplingConfig.fixed_interval(0.5, 0.0)  # 0.5 seconds
)
print(f"Range rate computer: {range_rate}")
# Range rate computer: RangeRateComputer()

# Create a simple scenario to demonstrate usage
# ISS orbit
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2, 60.0).with_name("ISS")

epoch_start = propagator.epoch
epoch_end = epoch_start + 24 * 3600.0  # 24 hours

# Ground station
location = bh.PointLocation(-74.0060, 40.7128, 0.0)

# Compute accesses with range rate
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
windows = bh.location_accesses(
    location,
    propagator,
    epoch_start,
    epoch_end,
    constraint,
    property_computers=[range_rate],
)

# Access computed properties
window = windows[0]
rr_data = window.properties.additional["range_rate"]
velocities_mps = rr_data["values"]  # m/s (positive=receding)
print(
    f"\nRange rate varies from {min(velocities_mps):.1f} to {max(velocities_mps):.1f} m/s"
)
print("Negative = approaching (decreasing distance)")
print("Positive = receding (increasing distance)")
# Range rate varies from -6382.0 to 6372.9 m/s
# Negative = approaching (decreasing distance)
# Positive = receding (increasing distance)
