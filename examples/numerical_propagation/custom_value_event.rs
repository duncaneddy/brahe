//! Creating custom ValueEvent detectors.
//! Demonstrates detecting when a computed value crosses a threshold.

use brahe as bh;
use bh::events::{DValueEvent, EventDirection};
use bh::traits::DStatePropagator;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,
        0.01,
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
        bh::ForceModelConfiguration::default(),
        Some(params),
        None,
        None,
        None,
    )
    .unwrap();

    // Create ValueEvent: detect when z crosses 0 (equator crossing)
    // Ascending node: z goes from negative to positive (INCREASING)
    let ascending_node_fn =
        |_t: bh::Epoch, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| state[2];
    let ascending_node = DValueEvent::new(
        "Ascending Node",
        ascending_node_fn, // z-component
        0.0,               // target value
        EventDirection::Increasing,
    );

    // Descending node: z goes from positive to negative (DECREASING)
    let descending_node_fn =
        |_t: bh::Epoch, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| state[2];
    let descending_node = DValueEvent::new(
        "Descending Node",
        descending_node_fn,
        0.0,
        EventDirection::Decreasing,
    );

    prop.add_event_detector(Box::new(ascending_node));
    prop.add_event_detector(Box::new(descending_node));

    // Propagate for 3 orbits
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + 3.0 * orbital_period);

    // Check detected events
    let events = prop.event_log();
    let ascending: Vec<_> = events
        .iter()
        .filter(|e| e.name.contains("Ascending"))
        .collect();
    let descending: Vec<_> = events
        .iter()
        .filter(|e| e.name.contains("Descending"))
        .collect();

    println!("Equator crossings over 3 orbits:");
    println!("  Ascending nodes: {}", ascending.len());
    println!("  Descending nodes: {}", descending.len());

    for event in events.iter().take(6) {
        let dt = event.window_open - epoch;
        let z = event.entry_state[2];
        println!("  '{}' at t+{:.1}s (z={:.1} m)", event.name, dt, z);
    }

    // Validate
    assert_eq!(ascending.len(), 3); // One per orbit
    assert_eq!(descending.len(), 3); // One per orbit

    println!("\nExample validated successfully!");
}
