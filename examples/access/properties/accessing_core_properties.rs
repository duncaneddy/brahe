//! Access core properties from computed access windows

#[allow(unused_imports)]
use brahe as bh;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    // Create location (San Francisco area)
    let location = bh::PointLocation::new(
        -122.4194,
        37.7749,
        0.0
    );

    // Create propagator from TLE (ISS example)
    let tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999";
    let tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601";
    let propagator = bh::SGPPropagator::from_tle(tle_line1, tle_line2, 60.0)?;

    // Define time period (24 hours from epoch)
    let epoch_start = bh::Epoch::from_datetime(2025, 11, 2, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epoch_end = epoch_start + 86400.0;

    // Create elevation constraint
    let constraint = bh::ElevationConstraint::new(Some(10.0), None)?;

    // Compute access windows
    let windows = bh::location_accesses(
        &location,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        None,
        None,
        None,
    );

    // Access core properties from first window
    if !windows.is_empty() {
        let window = &windows[0];
        let props = &window.properties;

        println!("Window ");
        let t_start = window.window_open;
        let t_end = window.window_close;
        println!("  Start: {}", t_start);
        println!("  End:   {}", t_end);
        println!("  Duration: {:.1} seconds", window.duration());
        println!("  Midtime: {}", window.midtime());

        println!("\nProperties:");

        // Azimuth values (open and close)
        let az_open = props.azimuth_open;
        let az_close = props.azimuth_close;
        println!("  Azimuth - Min: {:.1}°, Max: {:.1}°", az_open, az_close);

        // Elevation range (min and max)
        let elev_min = props.elevation_min;
        let elev_max = props.elevation_max;
        println!("  Elevation - Min: {:.1}°, Max: {:.1}°", elev_min, elev_max);

        // Off-nadir range (min and max)
        let off_nadir_min = props.off_nadir_min;
        let off_nadir_max = props.off_nadir_max;
        println!("  Off-nadir - Min: {:.1}°, Max: {:.1}°", off_nadir_min, off_nadir_max);

        // Local solar time at midpoint (local_time is in seconds since midnight)
        let local_time = props.local_time;
        let hours = (local_time / 3600.0).floor() as i32;
        let minutes = (local_time - hours as f64 * 3600.0) / 60.0;
        println!("  Local time: {:02}:{:05.2}", hours, minutes);

        // Look direction
        let look = &props.look_direction;
        println!("  Look direction: {}", look);

        // Ascending/Descending
        let asc_dsc = &props.asc_dsc;
        println!("  Ascending/Descending: {}", asc_dsc);
    }

    Ok(())
}
// Expected output (values will vary based on TLE and time):
// Window 
//   Start: 2025-11-02 05:39:28.345 UTC
//   End:   2025-11-02 05:44:00.000 UTC
//   Duration: 271.7 seconds
//   Midtime: 2025-11-02 05:41:44.172 UTC

// Properties:
//   Azimuth - Min: 177.0°, Max: 87.3°
//   Elevation - Min: 10.0°, Max: 18.7°
//   Off-nadir - Min: 62.6°, Max: 67.4°
//   Local time: 05:37.24
//   Look direction: Left
//   Ascending/Descending: Ascending
