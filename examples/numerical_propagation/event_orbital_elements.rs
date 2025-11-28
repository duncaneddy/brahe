//! Using orbital element events to detect inclination value crossings.
//! Demonstrates InclinationEvent with the angle_format parameter.

use brahe as bh;
use bh::events::{DInclinationEvent, DSemiMajorAxisEvent, EventDirection};
use bh::traits::DStatePropagator;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state - SSO-like orbit
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 600e3, 0.001, 97.8, 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let params = na::DVector::from_vec(vec![500.0, 2.0, 2.2, 2.0, 1.3]);

    // Create propagator
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfiguration::default(),
        Some(params),
        None,
        None,
        None,
    )
    .unwrap();

    // Add orbital element events
    // Detect when inclination crosses 97.79 degrees (monitoring for stability)
    let inc_event = DInclinationEvent::new(
        97.79, // value in degrees
        "Inc value".to_string(),
        EventDirection::Any,
        bh::AngleFormat::Degrees,
    );

    // Detect semi-major axis value (orbit decay monitoring)
    let sma_event = DSemiMajorAxisEvent::new(
        bh::R_EARTH + 599.5e3, // value in meters
        "SMA value".to_string(),
        EventDirection::Decreasing,
    );

    prop.add_event_detector(Box::new(inc_event));
    prop.add_event_detector(Box::new(sma_event));

    // Propagate for 3 orbits
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + 3.0 * orbital_period);

    // Check detected events
    let events = prop.event_log();
    println!("Detected {} orbital element events:", events.len());

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
    let inc_events: Vec<_> = events.iter().filter(|e| e.name.contains("Inc")).collect();
    let sma_events: Vec<_> = events.iter().filter(|e| e.name.contains("SMA")).collect();

    println!("\nInclination value crossings: {}", inc_events.len());
    println!("SMA value crossings: {}", sma_events.len());

    // The J2 perturbation causes slow variations - we may or may not cross values
    // depending on the exact parameters, so we just validate the events work
    println!("\nExample completed successfully!");
}
