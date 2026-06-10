#[allow(unused_imports)]
use brahe as bh;
use brahe::utils::Identifiable;

fn main() {
    // Initialize EOP
    bh::initialize_eop().unwrap();

    // Set the location
    let location = bh::PointLocation::new(
        -122.4194,      // Longitude [deg]
        37.7749,        // Latitude [deg]
        0.0             // Altitude [m]
    )
        .with_name("San Francisco");

    // Get the latest TLE for the ISS (NORAD ID 25544) from Celestrak
    let client = bh::celestrak::CelestrakClient::new();
    let propagator = client.get_sgp_propagator_by_catnr(25544, 60.0).unwrap();

    // Configure Search Window
    let epoch_start = bh::Epoch::now();
    let epoch_end = epoch_start + 7.0 * 86400.0;  // 7 days later

    // Set access constraints -> Must be above 10 degrees elevation
    let constraint = bh::ElevationConstraint::new(Some(10.0), None).unwrap();

    // Compute access windows
    let windows = bh::location_accesses(
        &location,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        None,
        None,
    ).unwrap();

    // Print first 3 access windows
    for window in windows.iter().take(3) {
        println!(
            "Access Window: {} to {}, Duration: {:.2} minutes",
            window.window_open,
            window.window_close,
            window.duration() / 60.0
        );
    }
}

