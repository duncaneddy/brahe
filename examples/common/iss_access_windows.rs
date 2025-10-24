//! This example shows how to find passes of the ISS over San Francisco, CA
//! using an elevation constraint.

#[allow(unused_imports)]
use brahe as bh;
use brahe::{Epoch, PointLocation, ElevationConstraint, location_accesses};
use brahe::datasets::celestrak::get_tle_by_id_as_propagator;
use brahe::utils::Identifiable;

fn main() {
    // Initialize EOP
    bh::initialize_eop().unwrap();

    // Set the location
    let location = PointLocation::new(-122.4194, 37.7749, 0.0)
        .with_name("San Francisco");

    // Get the latest TLE for the ISS (NORAD ID 25544) from Celestrak
    let propagator = get_tle_by_id_as_propagator(25544, None, 60.0).unwrap();

    // Configure Search Window
    let epoch_start = Epoch::now();
    let epoch_end = epoch_start + 7.0 * 86400.0;  // 7 days later

    // Set access constraints -> Must be above 10 degrees elevation
    let constraint = ElevationConstraint::new(Some(10.0), None).unwrap();

    // Compute access windows
    let windows = location_accesses(
        &location,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        None,
        None,
        None
    );

    assert!(!windows.is_empty(), "Should find at least one access window");

    // Print first 3 access windows
    for window in windows.iter().take(3) {
        println!(
            "Access Window: {} to {}, Duration: {:.2} minutes",
            window.window_open,
            window.window_close,
            window.duration() / 60.0
        );
    }
    // Outputs (will vary based on current time and ISS orbit):
    // Access Window: 2025-10-25 08:49:40.062 UTC to 2025-10-25 08:53:48.463 UTC, Duration: 4.14 minutes
    // Access Window: 2025-10-25 10:25:40.245 UTC to 2025-10-25 10:31:48.463 UTC, Duration: 6.14 minutes
    // Access Window: 2025-10-25 12:05:33.455 UTC to 2025-10-25 12:06:48.463 UTC, Duration: 1.25 minutes
}
