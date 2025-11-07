//! ```cargo
//! [dependencies]
//! brahe = { path = "../../.." }
//! ```

#[allow(unused_imports)]
use brahe as bh;
use bh::access::{RangeComputer, SamplingConfig, PropertyValue};
use bh::utils::Identifiable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    // Compute range at 50 evenly-spaced points
    let range_comp = RangeComputer::new(
        SamplingConfig::FixedCount(50)
    );
    println!("Range computer: sampling=FixedCount(50)");
    // Range computer: sampling=FixedCount(50)

    // ISS orbit
    let tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999";
    let tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601";
    let propagator = bh::SGPPropagator::from_tle(tle_line1, tle_line2, 60.0)?
        .with_name("ISS");

    let epoch_start = propagator.epoch;
    let epoch_end = epoch_start + 24.0 * 3600.0;

    // Ground station
    let location = bh::PointLocation::new(-74.0060, 40.7128, 0.0);

    // Compute accesses with range
    let constraint = bh::ElevationConstraint::new(Some(10.0), None)?;
    let windows = bh::location_accesses(
        &location,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        Some(&[&range_comp]),  // Property computers
        None,  // Use default config
        None,  // No time tolerance
    )?;

    // Access computed properties
    let window = &windows[0];
    let range_data = window.properties.additional.get("range").unwrap();

    // Extract values from TimeSeries
    let distances_m = match range_data {
        PropertyValue::TimeSeries { values, .. } => values,
        _ => panic!("Expected TimeSeries"),
    };

    // Convert to km
    let min_km = distances_m.iter().fold(f64::INFINITY, |a: f64, &b| a.min(b)) / 1000.0;
    let max_km = distances_m.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b)) / 1000.0;

    println!("\nRange varies from {:.1} to {:.1} km", min_km, max_km);
    // Range varies from 658.9 to 1501.0 km

    Ok(())
}
