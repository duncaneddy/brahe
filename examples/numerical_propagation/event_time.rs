//! Using TimeEvent for scheduled event detection.
//! Demonstrates triggering events at specific times during propagation.

use brahe as bh;
use bh::events::DTimeEvent;
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,
        0.001,
        97.8,
        15.0,
        30.0,
        45.0,
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

    // Add time events at specific epochs
    let event_30min = DTimeEvent::new(epoch + 1800.0, "30-minute mark".to_string());
    let event_1hr = DTimeEvent::new(epoch + 3600.0, "1-hour mark".to_string());

    // Add a terminal event that stops propagation
    let event_terminal =
        DTimeEvent::new(epoch + 5400.0, "90-minute stop".to_string()).set_terminal();

    prop.add_event_detector(Box::new(event_30min));
    prop.add_event_detector(Box::new(event_1hr));
    prop.add_event_detector(Box::new(event_terminal));

    // Propagate for 2 hours (will stop at 90 minutes due to terminal event)
    prop.propagate_to(epoch + 7200.0);

    // Check detected events
    let events = prop.event_log();
    println!("Detected {} events:", events.len());
    for event in events {
        let dt = event.window_open - epoch;
        println!("  '{}' at t+{:.1}s", event.name, dt);
    }

    // Verify propagation stopped at terminal event
    let final_time = prop.current_epoch() - epoch;
    println!(
        "\nPropagation stopped at: t+{:.1}s (requested: t+7200s)",
        final_time
    );

    // Validate
    assert_eq!(events.len(), 3); // All three events detected
    assert!((final_time - 5400.0).abs() < 1.0); // Stopped at 90 min

    println!("\nExample validated successfully!");
}
