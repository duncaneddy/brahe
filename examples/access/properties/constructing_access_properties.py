# /// script
# dependencies = ["brahe"]
# ///
"""
Constructing AccessProperties directly using the builder API.

AccessProperties is normally produced by an access search, but it can also
be constructed directly -- for tests or custom pipelines that feed
geometry from another source. AccessProperties.builder() returns a
builder with every field unset; each of the 15 fields is set through a
chained named setter, and build() raises an error naming any field left
unset instead of applying a default.
"""

import brahe as bh

props = (
    bh.AccessProperties.builder()
    .azimuth_open(45.0)
    .azimuth_close(135.0)
    .elevation_min(10.0)
    .elevation_max(85.0)
    .elevation_open(12.0)
    .elevation_close(10.5)
    .off_nadir_min(5.0)
    .off_nadir_max(80.0)
    .local_time(43200.0)
    .look_direction(bh.LookDirection.RIGHT)
    .asc_dsc(bh.AscDsc.ASCENDING)
    .center_lon(0.0)
    .center_lat(45.0)
    .center_alt(0.0)
    .center_ecef([4517.59e3, 4517.59e3, 0.0])
    .build()
)

print(f"Azimuth open/close: {props.azimuth_open}, {props.azimuth_close} deg")
print(f"Elevation min/max: {props.elevation_min}, {props.elevation_max} deg")

# The flat constructor takes the same 15 fields positionally, without
# naming each one -- an alternative when all values are already at hand.
props_flat = bh.AccessProperties(
    45.0,
    135.0,
    10.0,
    85.0,
    12.0,
    10.5,
    5.0,
    80.0,
    43200.0,
    bh.LookDirection.RIGHT,
    bh.AscDsc.ASCENDING,
    0.0,
    45.0,
    0.0,
    [4517.59e3, 4517.59e3, 0.0],
)

assert props_flat.azimuth_open == props.azimuth_open
print("Example validated successfully!")
