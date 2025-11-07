//! ```cargo
//! [dependencies]
//! brahe = { path = "../../.." }
//! ```

#[allow(unused_imports)]
use brahe as bh;
use bh::access::{DopplerComputer, SamplingConfig, PropertyValue};
use bh::utils::Identifiable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    // S-band downlink only (8.4 GHz)
    let _doppler = DopplerComputer::new(
        None,  // No uplink
        Some(8.4e9),  // Downlink frequency
        SamplingConfig::FixedInterval { interval: 0.1, offset: 0.0 }  // 0.1 seconds
    );
    println!("Downlink only: uplink=None, downlink=8.4e9 Hz");
    // Downlink only: uplink=None, downlink=8.4e9 Hz

    // Both uplink (2.0 GHz) and downlink (8.4 GHz)
    let doppler = DopplerComputer::new(
        Some(2.0e9),  // Uplink frequency
        Some(8.4e9),  // Downlink frequency
        SamplingConfig::FixedCount(100)
    );
    println!("Both frequencies: uplink=2.0e9 Hz, downlink=8.4e9 Hz");
    // Both frequencies: uplink=2.0e9 Hz, downlink=8.4e9 Hz

    // ISS orbit
    let tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999";
    let tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601";
    let propagator = bh::SGPPropagator::from_tle(tle_line1, tle_line2, 60.0)?
        .with_name("ISS");

    let epoch_start = propagator.epoch;
    let epoch_end = epoch_start + 24.0 * 3600.0;

    // Ground station (lon, lat, alt)
    let location = bh::PointLocation::new(-74.0060, 40.7128, 0.0);

    // Compute accesses with Doppler
    let constraint = bh::ElevationConstraint::new(Some(10.0), None)?;
    let windows = bh::location_accesses(
        &location,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        Some(&[&doppler]),  // Property computers
        None,  // Use default config
        None,  // No time tolerance
    )?;

    // Access computed properties
    let window = &windows[0];
    let doppler_data = window.properties.additional.get("doppler_downlink").unwrap();

    // Extract values from TimeSeries
    let values = match doppler_data {
        PropertyValue::TimeSeries { values, .. } => values,
        _ => panic!("Expected TimeSeries"),
    };

    let min_val = values.iter().fold(f64::INFINITY, |a: f64, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));

    println!("\nFirst pass downlink Doppler shift range: {:.1} to {:.1} Hz", min_val, max_val);
    // First pass Doppler shift range: -189220.9 to 189239.8 Hz

    Ok(())
}
