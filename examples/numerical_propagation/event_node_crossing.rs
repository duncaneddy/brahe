//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.34"
//! ```
//!
//! Using node crossing events to detect equatorial crossings.
//! Demonstrates detecting ascending and descending node passages.

use brahe as bh;
use bh::events::{DAscendingNodeEvent, DDescendingNodeEvent};
use bh::traits::DStatePropagator;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    bh::initialize_eop().unwrap();

    // Create initial epoch and state - inclined orbit
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    // Inclined orbit for clear node crossings
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0);
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

    // Add node crossing events
    // Ascending node: spacecraft crosses equator heading north (argument of latitude = 0)
    let asc_event = DAscendingNodeEvent::new("Ascending Node");

    // Descending node: spacecraft crosses equator heading south (argument of latitude = 180 deg)
    let desc_event = DDescendingNodeEvent::new("Descending Node");

    prop.add_event_detector(Box::new(asc_event));
    prop.add_event_detector(Box::new(desc_event));

    // Propagate for 3 orbits
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + 3.0 * orbital_period);

    // Check detected events
    let events = prop.event_log();
    println!("Detected {} node crossing events:", events.len());

    for event in events {
        let dt = event.window_open - epoch;
        // Compute geodetic latitude at event
        let r_eci = na::Vector3::new(
            event.entry_state[0],
            event.entry_state[1],
            event.entry_state[2],
        );
        let r_ecef = bh::position_eci_to_ecef(event.window_open, r_eci);
        let geodetic = bh::position_ecef_to_geodetic(r_ecef, bh::AngleFormat::Degrees);
        let lat = geodetic[1];
        println!(
            "  '{}' at t+{:.1}s (latitude: {:.2} deg)",
            event.name, dt, lat
        );
    }

    // Count events by type
    let ascending: Vec<_> = events
        .iter()
        .filter(|e| e.name.contains("Ascending"))
        .collect();
    let descending: Vec<_> = events
        .iter()
        .filter(|e| e.name.contains("Descending"))
        .collect();

    println!("\nAscending node crossings: {}", ascending.len());
    println!("Descending node crossings: {}", descending.len());

    // Validate
    assert!(
        ascending.len() >= 3,
        "Expected at least 3 ascending node crossings"
    );
    assert!(
        descending.len() >= 3,
        "Expected at least 3 descending node crossings"
    );
}

// Expected output:
// Detected 6 node crossing events:
//   'Descending Node' at t+2831.5s (latitude: -0.13 deg)
//   'Ascending Node' at t+5663.2s (latitude: 0.13 deg)
//   'Descending Node' at t+8494.6s (latitude: -0.13 deg)
//   'Ascending Node' at t+11326.4s (latitude: 0.13 deg)
//   'Descending Node' at t+14157.5s (latitude: -0.13 deg)
//   'Ascending Node' at t+16989.6s (latitude: 0.13 deg)
//
// Ascending node crossings: 3
// Descending node crossings: 3
