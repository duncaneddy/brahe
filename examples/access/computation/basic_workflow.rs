//! Basic access computation workflow: finding satellite passes over a ground location
// TIMEOUT = 600

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize EOP
    bh::initialize_eop()?;

    // Define ground location
    let location = bh::PointLocation::new(
        -122.4194,
        37.7749,
        0.0,
    )
    .with_name("San Francisco");

    // Create propagator from TLE
    let tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999";
    let tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601";
    let propagator = bh::SGPPropagator::from_tle(tle_line1, tle_line2, 60.0)?
        .with_name("ISS");

    // Define time window
    let epoch_start = bh::Epoch::from_datetime(2025, 11, 2, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epoch_end = epoch_start + 7.0 * 86400.0;

    // Define constraint
    let constraint = bh::ElevationConstraint::new(Some(10.0), None)?;

    // Compute access windows
    let windows = bh::location_accesses(
        &location,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        None, // No custom property computers
        None, // Use default config
    )?;

    // Process results
    println!("Found {} access windows", windows.len());
    for (i, window) in windows.iter().take(3).enumerate() {
        let duration_min = window.duration() / 60.0;
        println!("\nWindow {}:", i + 1);
        println!("  Start: {}", window.window_open);
        println!("  End:   {}", window.window_close);
        println!("  Duration: {:.2} minutes", duration_min);

        // Access computed properties
        let elev_max = window.properties.elevation_max;
        println!("  Max elevation: {:.1}°", elev_max);
    }

    Ok(())
}

