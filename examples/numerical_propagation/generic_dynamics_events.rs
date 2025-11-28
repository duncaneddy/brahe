//! Event detection with NumericalPropagator.
//! Demonstrates detecting zero crossings in a simple harmonic oscillator.

use brahe as bh;
use bh::events::{DValueEvent, EventDirection};
use bh::traits::DStatePropagator;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data (needed for epoch operations)
    bh::initialize_eop().unwrap();

    // Create initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Simple Harmonic Oscillator
    // State: [x, v] where x is position and v is velocity
    let omega = 2.0 * PI; // 1 Hz oscillation frequency

    // Initial state: displaced from equilibrium
    let x0 = 1.0; // 1 meter displacement
    let v0 = 0.0; // Starting from rest
    let initial_state = na::DVector::from_vec(vec![x0, v0]);

    // SHO dynamics function
    let dynamics_fn: bh::DStateDynamics = Box::new(
        move |_t: f64, state: &na::DVector<f64>, params: Option<&na::DVector<f64>>| {
            let x = state[0];
            let v = state[1];
            let omega_sq = params.map(|p| p[0]).unwrap_or(omega * omega);
            na::DVector::from_vec(vec![v, -omega_sq * x])
        },
    );

    // Parameters (omega^2)
    let params = na::DVector::from_vec(vec![omega * omega]);

    // Create propagator
    let mut prop = bh::DNumericalPropagator::new(
        epoch,
        initial_state,
        dynamics_fn,
        bh::NumericalPropagationConfig::default(),
        Some(params),
        None, // No control input
        None, // No initial covariance
    )
    .unwrap();

    // Create ValueEvent to detect position zero crossings
    // INCREASING: x goes from negative to positive (moving right through origin)
    let positive_fn =
        |_t: bh::Epoch, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| state[0];
    let positive_crossing = DValueEvent::new(
        "Positive Crossing",
        positive_fn,
        0.0, // Target value
        EventDirection::Increasing,
    );

    // DECREASING: x goes from positive to negative (moving left through origin)
    let negative_fn =
        |_t: bh::Epoch, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| state[0];
    let negative_crossing = DValueEvent::new(
        "Negative Crossing",
        negative_fn,
        0.0,
        EventDirection::Decreasing,
    );

    // Add event detectors to propagator
    prop.add_event_detector(Box::new(positive_crossing));
    prop.add_event_detector(Box::new(negative_crossing));

    // Propagate for 5 periods
    let period = 2.0 * PI / omega; // Period = 1 second
    prop.propagate_to(epoch + 5.0 * period);

    // Get event log
    let events = prop.event_log();

    println!("Simple Harmonic Oscillator Zero Crossings:");
    println!("  omega = {:.4} rad/s (1 Hz)", omega);
    println!("  Period = {:.4} s", period);
    println!("  Expected crossings per period: 2 (one each direction)");
    println!();

    let positive_events: Vec<_> = events
        .iter()
        .filter(|e| e.name.contains("Positive"))
        .collect();
    let negative_events: Vec<_> = events
        .iter()
        .filter(|e| e.name.contains("Negative"))
        .collect();

    println!("Total events detected: {}", events.len());
    println!("  Positive crossings: {}", positive_events.len());
    println!("  Negative crossings: {}", negative_events.len());
    println!();

    println!("Event details:");
    println!("  Time (s)   Type               Position     Velocity");
    println!("{}", "-".repeat(60));

    for event in events.iter().take(10) {
        let t = event.window_open - epoch;
        let x = event.entry_state[0];
        let v = event.entry_state[1];
        println!(
            "  {:.4}     {:<18} {:+.6}   {:+.6}",
            t, event.name, x, v
        );
    }

    // Validate
    // In 5 periods, we should have 5 positive crossings and 5 negative crossings
    assert_eq!(
        positive_events.len(),
        5,
        "Expected 5 positive crossings, got {}",
        positive_events.len()
    );
    assert_eq!(
        negative_events.len(),
        5,
        "Expected 5 negative crossings, got {}",
        negative_events.len()
    );

    // Check timing: crossings should occur at quarter periods
    // Starting from x=1, v=0: oscillator moves left first (cosine motion)
    // Negative crossing (moving left) at T/4, 5T/4, 9T/4, ...
    // Positive crossing (moving right) at 3T/4, 7T/4, 11T/4, ...
    let expected_negative_times: Vec<f64> = (0..5).map(|i| (0.25 + i as f64) * period).collect();
    let expected_positive_times: Vec<f64> = (0..5).map(|i| (0.75 + i as f64) * period).collect();

    for (i, event) in negative_events.iter().enumerate() {
        let t = event.window_open - epoch;
        let expected = expected_negative_times[i];
        let error = (t - expected).abs();
        assert!(
            error < 0.02,
            "Negative crossing {}: expected t={:.4}, got t={:.4}",
            i,
            expected,
            t
        );
    }

    for (i, event) in positive_events.iter().enumerate() {
        let t = event.window_open - epoch;
        let expected = expected_positive_times[i];
        let error = (t - expected).abs();
        assert!(
            error < 0.02,
            "Positive crossing {}: expected t={:.4}, got t={:.4}",
            i,
            expected,
            t
        );
    }

    println!("\nTiming verified: all crossings within 0.02s of expected times");

    println!("\nExample validated successfully!");
}
