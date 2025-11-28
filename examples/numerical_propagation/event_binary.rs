//! Using BinaryEvent for boolean condition detection.
//! Demonstrates detecting when a condition transitions between true and false.

use brahe as bh;
use bh::events::{DBinaryEvent, EdgeType};
use bh::traits::DStatePropagator;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Create binary events for hemisphere crossings
    // Rising edge: false → true (entering northern hemisphere)
    let enter_north = DBinaryEvent::new(
        "Enter Northern",
        |_t, state: &na::DVector<f64>, _params| {
            // Returns True if z-position is positive (northern hemisphere)
            state[2] > 0.0
        },
        EdgeType::RisingEdge,
    );

    // Falling edge: true → false (exiting northern hemisphere)
    let exit_north = DBinaryEvent::new(
        "Exit Northern",
        |_t, state: &na::DVector<f64>, _params| state[2] > 0.0,
        EdgeType::FallingEdge,
    );

    // Create propagator
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfig::two_body_gravity(),
        None,
        None,
        None,
        None,
    )
    .unwrap();

    prop.add_event_detector(Box::new(enter_north));
    prop.add_event_detector(Box::new(exit_north));

    // Propagate for 2 orbits
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + 2.0 * orbital_period);

    // Check detected events
    let events = prop.event_log();
    let enters: Vec<_> = events.iter().filter(|e| e.name.contains("Enter")).collect();
    let exits: Vec<_> = events.iter().filter(|e| e.name.contains("Exit")).collect();

    println!("Hemisphere crossings over 2 orbits:");
    println!("  Entered northern: {} times", enters.len());
    println!("  Exited northern:  {} times", exits.len());

    println!("\nEvent timeline:");
    for event in events.iter().take(8) {
        let dt = event.window_open - epoch;
        let z_km = event.entry_state[2] / 1e3;
        println!(
            "  t+{:7.1}s: {:16} (z = {:+.1} km)",
            dt, event.name, z_km
        );
    }

    // Validate
    assert_eq!(enters.len(), 2); // Once per orbit
    assert_eq!(exits.len(), 2); // Once per orbit

    println!("\nExample validated successfully!");
}
