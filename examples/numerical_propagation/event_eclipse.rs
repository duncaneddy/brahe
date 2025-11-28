//! Using eclipse events to detect shadow transitions.
//! Demonstrates detecting when a spacecraft enters or exits Earth's shadow.

use brahe as bh;
use bh::events::{DEclipseEvent, EdgeType};
use bh::traits::DStatePropagator;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    bh::initialize_eop().unwrap();

    // Create initial epoch and state - LEO orbit
    let epoch = bh::Epoch::from_datetime(2024, 6, 21, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    // LEO orbit with some inclination
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

    // Add eclipse events with different edge types
    // Detect entry into eclipse (any shadow - umbra or penumbra)
    let eclipse_entry = DEclipseEvent::new("Eclipse Entry", EdgeType::RisingEdge, None);

    // Detect exit from eclipse
    let eclipse_exit = DEclipseEvent::new("Eclipse Exit", EdgeType::FallingEdge, None);

    prop.add_event_detector(Box::new(eclipse_entry));
    prop.add_event_detector(Box::new(eclipse_exit));

    // Propagate for 5 orbits
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    let _ = prop.propagate_to(epoch + 5.0 * orbital_period);

    // Check detected events
    let events = prop.event_log();
    println!("Detected {} eclipse events:", events.len());

    for event in events {
        let dt = event.window_open - epoch;
        println!("  '{}' at t+{:.1}s", event.name, dt);
    }

    // Count events by type
    let entries: Vec<_> = events.iter().filter(|e| e.name.contains("Entry")).collect();
    let exits: Vec<_> = events.iter().filter(|e| e.name.contains("Exit")).collect();

    println!("\nEclipse entries: {}", entries.len());
    println!("Eclipse exits: {}", exits.len());

    // Calculate eclipse durations
    if !entries.is_empty() && !exits.is_empty() {
        let mut durations: Vec<f64> = Vec::new();
        for entry in &entries {
            // Find next exit after this entry
            for exit_event in &exits {
                if exit_event.window_open > entry.window_open {
                    let duration = exit_event.window_open - entry.window_open;
                    durations.push(duration);
                    break;
                }
            }
        }

        if !durations.is_empty() {
            let avg_duration: f64 = durations.iter().sum::<f64>() / durations.len() as f64;
            println!(
                "\nAverage eclipse duration: {:.1}s ({:.1} min)",
                avg_duration,
                avg_duration / 60.0
            );
        }
    }

    // Validate - should have roughly equal entries and exits
    assert!(
        (entries.len() as i32 - exits.len() as i32).abs() <= 1,
        "Entry/exit count mismatch"
    );
    assert!(
        entries.len() >= 4,
        "Expected at least 4 eclipse entries in 5 orbits"
    );

    println!("\nExample validated successfully!");
}

// Expected output:
// Detected 10 eclipse events:
//   'Eclipse Entry' at t+1923.4s
//   'Eclipse Exit' at t+4078.2s
//   'Eclipse Entry' at t+7508.5s
//   'Eclipse Exit' at t+9663.3s
//   'Eclipse Entry' at t+13093.6s
//   'Eclipse Exit' at t+15248.4s
//   'Eclipse Entry' at t+18678.7s
//   'Eclipse Exit' at t+20833.5s
//   'Eclipse Entry' at t+24263.8s
//   'Eclipse Exit' at t+26418.6s
//
// Eclipse entries: 5
// Eclipse exits: 5
//
// Average eclipse duration: 2154.9s (35.9 min)
//
// Example validated successfully!
