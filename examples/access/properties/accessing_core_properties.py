# /// script
# dependencies = ["brahe"]
# ///
"""
Access core properties from computed access windows
"""

import brahe as bh

bh.initialize_eop()

# Create location (San Francisco area)
location = bh.PointLocation(-122.4194, 37.7749, 0.0)

# Create propagator from TLE (ISS example)
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2, 60.0)

# Define time period (24 hours from epoch)
epoch_start = bh.Epoch.from_datetime(2025, 11, 2, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
epoch_end = epoch_start + 86400.0

# Create elevation constraint
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

# Compute access windows
windows = bh.location_accesses(
    [location], [propagator], epoch_start, epoch_end, constraint
)

# Access core properties from first window
if windows:
    window = windows[0]
    props = window.properties

    print("Window ")
    t_start = window.window_open
    t_end = window.window_close
    print(f"  Start: {t_start}")
    print(f"  End:   {t_end}")
    print(f"  Duration: {window.duration:.1f} seconds")
    print(f"  Midtime: {window.midtime}")

    print("\nProperties:")

    # Azimuth values (open and close)
    az_open = props.azimuth_open
    az_close = props.azimuth_close
    print(f"  Azimuth - Min: {az_open:.1f}°, Max: {az_close:.1f}°")

    # Elevation range (min and max)
    elev_min = props.elevation_min
    elev_max = props.elevation_max
    print(f"  Elevation - Min: {elev_min:.1f}°, Max: {elev_max:.1f}°")

    # Off-nadir range (min and max)
    off_nadir_min = props.off_nadir_min
    off_nadir_max = props.off_nadir_max
    print(f"  Off-nadir - Min: {off_nadir_min:.1f}°, Max: {off_nadir_max:.1f}°")

    # Local solar time at midpoint
    local_time = props.local_time
    hours = int(local_time // 3600)
    minutes = (local_time - hours * 3600) / 60
    print(f"  Local time: {hours:02d}:{minutes:02.2f}")

    # Look direction
    look = props.look_direction
    print(f"  Look direction: {look}")

    # Ascending/Descending
    asc_dsc = props.asc_dsc
    print(f"  Ascending/Descending: {asc_dsc}")

# Expected output (values will vary based on TLE and time):
# Window
#   Start: 2025-11-02 05:39:28.345 UTC
#   End:   2025-11-02 05:44:00.000 UTC
#   Duration: 271.7 seconds
#   Midtime: 2025-11-02 05:41:44.172 UTC

# Properties:
#   Azimuth - Min: 177.0°, Max: 87.3°
#   Elevation - Min: 10.0°, Max: 18.7°
#   Off-nadir - Min: 62.6°, Max: 67.4°
#   Local time: 05:37.24
#   Look direction: Left
#   Ascending/Descending: Ascending
