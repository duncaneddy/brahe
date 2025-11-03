//! Configure access search algorithm with custom parameters for performance tuning

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    // Create custom configuration
    let config = bh::AccessSearchConfig {
        initial_time_step: 60.0,
        adaptive_step: true,
        adaptive_fraction: 0.75,
        parallel: true,
        num_threads: Some(0),
    };

    // Use custom config with location and propagator
    let location = bh::PointLocation::new(
       -122.4194,
       37.7749,
        0.0,
    )
    .with_name("San Francisco");

    let tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999";
    let tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601";
    let propagator = bh::SGPPropagator::from_tle(tle_line1, tle_line2, 60.0)?
        .with_name("ISS");

    let epoch_start = bh::Epoch::from_datetime(2025, 11, 2, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epoch_end = epoch_start + 86400.0;
    let constraint = bh::ElevationConstraint::new(Some(10.0), None)?;

    let windows = bh::location_accesses(
        &location,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        None,
        Some(&config),
        None,
    );

    println!(
        "Found {} access windows with custom configuration",
        windows.len()
    );
    println!(
        "Configuration: {}s time step, adaptive={}",
        config.initial_time_step, config.adaptive_step
    );

    Ok(())
}

// Output:
// Found 5 access windows with custom configuration
// Configuration: 60s time step, adaptive=true
