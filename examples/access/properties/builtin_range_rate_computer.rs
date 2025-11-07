//! ```cargo
//! [dependencies]
//! brahe = { path = "../../.." }
//! ```

#[allow(unused_imports)]
use brahe as bh;
use bh::access::{RangeRateComputer, SamplingConfig, PropertyValue};
use bh::utils::Identifiable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    // Compute range rate every 0.5 seconds
    let range_rate = RangeRateComputer::new(
        SamplingConfig::FixedInterval { interval: 0.5, offset: 0.0 }  // 0.5 seconds
    );
    println!("Range rate computer: sampling=FixedInterval(0.5s)");
    // Range rate computer: sampling=FixedInterval(0.5s)

    // ISS orbit
    let tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999";
    let tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601";
    let propagator = bh::SGPPropagator::from_tle(tle_line1, tle_line2, 60.0)?
        .with_name("ISS");

    let epoch_start = propagator.epoch;
    let epoch_end = epoch_start + 24.0 * 3600.0;

    // Ground station
    let location = bh::PointLocation::new(-74.0060, 40.7128, 0.0);

    // Compute accesses with range rate
    let constraint = bh::ElevationConstraint::new(Some(10.0), None)?;
    let windows = bh::location_accesses(
        &location,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        Some(&[&range_rate]),  // Property computers
        None,  // Use default config
        None,  // No time tolerance
    )?;

    // Access computed properties
    let window = &windows[0];
    let rr_data = window.properties.additional.get("range_rate").unwrap();

    // Extract values from TimeSeries
    let velocities_mps = match rr_data {
        PropertyValue::TimeSeries { values, .. } => values,
        _ => panic!("Expected TimeSeries"),
    };

    let min_vel = velocities_mps.iter().fold(f64::INFINITY, |a: f64, &b| a.min(b));
    let max_vel = velocities_mps.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));

    println!("\nRange rate varies from {:.1} to {:.1} m/s", min_vel, max_vel);
    println!("Negative = approaching (decreasing distance)");
    println!("Positive = receding (increasing distance)");
    // Range rate varies from -6382.0 to 6372.9 m/s
    // Negative = approaching (decreasing distance)
    // Positive = receding (increasing distance)

    Ok(())
}
