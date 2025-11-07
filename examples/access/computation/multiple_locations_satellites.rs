//! Compute access windows for multiple ground locations and multiple satellites

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    // Define multiple locations
    let locations = vec![
        bh::PointLocation::new(
            -122.4194,
            37.7749,
            0.0,
        )
        .with_name("San Francisco"),
        bh::PointLocation::new(-71.0589, 42.3601, 0.0)
            .with_name("Boston"),
        bh::PointLocation::new(15.4038, 78.2232, 458.0)
            .with_name("Svalbard"),
    ];

    // Define multiple satellites
    let propagators = vec![
        bh::SGPPropagator::from_tle(
            "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999",
            "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601",
            60.0,
        )?
        .with_name("ISS"),
        bh::SGPPropagator::from_tle(
            "1 48274U 21035A   25306.17586037  .00031797  00000-0  38131-3 0  9995",
            "2 48274  41.4666 263.0710 0006682 308.7013  51.3228 15.60215133257694",
            60.0,
        )?
        .with_name("Tiangong"),
    ];

    // Compute access windows
    let epoch_start = bh::Epoch::from_datetime(2025, 11, 2, 2, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epoch_end = epoch_start + 86400.0;
    let constraint = bh::ElevationConstraint::new(Some(10.0), None)?;

    let windows = bh::location_accesses(
        &locations,
        &propagators,
        epoch_start,
        epoch_end,
        &constraint,
        None,
        None,
        None,
    )?;

    println!("Total windows: {}", windows.len());

    // Group by location
    let mut by_location: HashMap<String, Vec<&bh::AccessWindow>> = HashMap::new();
    for window in &windows {
        let loc_name = window.location_name.clone().unwrap_or_else(|| "Unknown".to_string());
        by_location
            .entry(loc_name)
            .or_insert_with(Vec::new)
            .push(window);
    }

    for (loc_name, loc_windows) in by_location {
        println!("\n{}: {} windows", loc_name, loc_windows.len());
    }

    Ok(())
}

// Expected output:
// Total windows: 20

// Boston: 10 windows

// San Francisco: 10 windows