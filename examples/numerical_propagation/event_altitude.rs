//! Using AltitudeEvent for altitude value detection.
//! Demonstrates detecting when altitude crosses a specified value.

use brahe as bh;
use bh::events::{DAltitudeEvent, EventDirection};
use bh::traits::DStatePropagator;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state - elliptical orbit
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    // Elliptical orbit: 300 km perigee, 800 km apogee
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 550e3,
        0.036,
        45.0,
        0.0,
        0.0,
        0.0,
    );
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let params = na::DVector::from_vec(vec![500.0, 2.0, 2.2, 2.0, 1.3]);

    // Create propagator
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfig::default(),
        Some(params),
        None,
        None,
        None,
    )
    .unwrap();

    // Add altitude events
    // Detect when crossing 500 km altitude (both directions)
    let event_500km = DAltitudeEvent::new(
        500e3, // value altitude in meters
        "500km crossing".to_string(),
        EventDirection::Any, // Detect both increasing and decreasing
    );

    // Detect only when ascending through 600 km
    let event_600km_up = DAltitudeEvent::new(
        600e3,
        "600km ascending".to_string(),
        EventDirection::Increasing,
    );

    prop.add_event_detector(Box::new(event_500km));
    prop.add_event_detector(Box::new(event_600km_up));

    // Propagate for 2 orbits
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + 2.0 * orbital_period);

    // Check detected events
    let events = prop.event_log();
    println!("Detected {} altitude events:", events.len());

    for event in events {
        let dt = event.window_open - epoch;
        let alt = event.entry_state.fixed_rows::<3>(0).norm() - bh::R_EARTH;
        println!(
            "  '{}' at t+{:.1}s (altitude: {:.1} km)",
            event.name,
            dt,
            alt / 1e3
        );
    }

    // Count events by type
    let crossings_500: Vec<_> = events
        .iter()
        .filter(|e| e.name.contains("500km"))
        .collect();
    let crossings_600: Vec<_> = events
        .iter()
        .filter(|e| e.name.contains("600km"))
        .collect();

    println!("\n500 km crossings (any direction): {}", crossings_500.len());
    println!("600 km ascending crossings: {}", crossings_600.len());

    // Validate
    assert!(crossings_500.len() >= 4); // At least 2 per orbit, 2 orbits
    assert!(crossings_600.len() >= 2); // At least 1 per orbit (ascending only)

    println!("\nExample validated successfully!");
}
