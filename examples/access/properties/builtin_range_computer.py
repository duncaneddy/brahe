# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using the built-in RangeComputer to compute slant range
"""

import brahe as bh

bh.initialize_eop()

# Compute range at 50 evenly-spaced points
range_comp = bh.RangeComputer(sampling_config=bh.SamplingConfig.fixed_count(50))
print(f"Range computer: {range_comp}")
# Range computer: RangeComputer(sampling=FixedCount(50))


# Create a simple scenario to demonstrate usage
# ISS orbit
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2, 60.0).with_name("ISS")

epoch_start = propagator.epoch
epoch_end = epoch_start + 24 * 3600.0  # 24 hours

# Ground station
location = bh.PointLocation(-74.0060, 40.7128, 0.0)

# Compute accesses with range
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
windows = bh.location_accesses(
    location,
    propagator,
    epoch_start,
    epoch_end,
    constraint,
    property_computers=[range_comp],
)

# Access computed properties
window = windows[0]
range_data = window.properties.additional["range"]
distances_m = range_data["values"]  # meters
distances_km = [d / 1000.0 for d in distances_m]
print(f"\nRange varies from {min(distances_km):.1f} to {max(distances_km):.1f} km")
# Range varies from 658.9 to 1501.0 km
