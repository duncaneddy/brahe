//! Event callback examples for logging and state inspection.
//! Demonstrates defining and attaching callbacks to event detectors.

use brahe as bh;
use bh::events::{DEventCallback, DTimeEvent, EventAction};
use bh::traits::DStatePropagator;
use nalgebra as na;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state - elliptical orbit
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Track callback invocations
    let callback_count = Arc::new(AtomicUsize::new(0));

    // Define a logging callback
    let count_ref = callback_count.clone();
    let logging_callback: DEventCallback = Box::new(
        move |event_epoch: bh::Epoch,
              event_state: &na::DVector<f64>,
              _params: Option<&na::DVector<f64>>|
              -> (
            Option<na::DVector<f64>>,
            Option<na::DVector<f64>>,
            EventAction,
        ) {
            let count = count_ref.fetch_add(1, Ordering::SeqCst) + 1;

            // Compute orbital elements at event time
            let state_vec = na::SVector::<f64, 6>::from_column_slice(event_state.as_slice());
            let koe = bh::state_eci_to_koe(state_vec, bh::AngleFormat::Degrees);
            let altitude = koe[0] - bh::R_EARTH;

            println!("  Event #{}:", count);
            println!("    Epoch: {}", event_epoch);
            println!("    Altitude: {:.1} km", altitude / 1e3);
            println!("    True anomaly: {:.1} deg", koe[5]);

            // Return unchanged state with CONTINUE action
            (None, None, EventAction::Continue)
        },
    );

    // Define another logging callback for second event
    let count_ref2 = callback_count.clone();
    let logging_callback2: DEventCallback = Box::new(
        move |event_epoch: bh::Epoch,
              event_state: &na::DVector<f64>,
              _params: Option<&na::DVector<f64>>|
              -> (
            Option<na::DVector<f64>>,
            Option<na::DVector<f64>>,
            EventAction,
        ) {
            let count = count_ref2.fetch_add(1, Ordering::SeqCst) + 1;

            let state_vec = na::SVector::<f64, 6>::from_column_slice(event_state.as_slice());
            let koe = bh::state_eci_to_koe(state_vec, bh::AngleFormat::Degrees);
            let altitude = koe[0] - bh::R_EARTH;

            println!("  Event #{}:", count);
            println!("    Epoch: {}", event_epoch);
            println!("    Altitude: {:.1} km", altitude / 1e3);
            println!("    True anomaly: {:.1} deg", koe[5]);

            (None, None, EventAction::Continue)
        },
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

    // Create time events with logging callbacks
    let event_log = DTimeEvent::new(epoch + 1000.0, "Log Event".to_string())
        .with_callback(logging_callback);
    prop.add_event_detector(Box::new(event_log));

    let event_log2 = DTimeEvent::new(epoch + 2000.0, "Log Event 2".to_string())
        .with_callback(logging_callback2);
    prop.add_event_detector(Box::new(event_log2));

    // Propagate for half an orbit
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    println!("Propagating with logging callbacks:");
    prop.propagate_to(epoch + orbital_period / 2.0);

    let final_count = callback_count.load(Ordering::SeqCst);
    println!("\nCallback invoked {} times", final_count);

    // Now demonstrate STOP action
    println!("\nDemonstrating STOP action:");

    // Define a callback that stops propagation
    let stop_callback: DEventCallback = Box::new(
        move |event_epoch: bh::Epoch,
              _event_state: &na::DVector<f64>,
              _params: Option<&na::DVector<f64>>|
              -> (
            Option<na::DVector<f64>>,
            Option<na::DVector<f64>>,
            EventAction,
        ) {
            println!("  Stopping at {}", event_epoch);
            (None, None, EventAction::Stop)
        },
    );

    let mut prop2 = bh::DNumericalOrbitPropagator::new(
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

    // Event that stops propagation at t+500s
    let stop_event =
        DTimeEvent::new(epoch + 500.0, "Stop Event".to_string()).with_callback(stop_callback);
    prop2.add_event_detector(Box::new(stop_event));

    // Try to propagate for one full orbit
    prop2.propagate_to(epoch + orbital_period);

    // Check where propagation actually stopped
    let actual_duration = prop2.current_epoch() - epoch;
    println!("  Requested duration: {:.1}s", orbital_period);
    println!("  Actual duration: {:.1}s", actual_duration);
    println!("  Stopped early: {}", actual_duration < orbital_period);

    // Validate
    assert_eq!(final_count, 2);
    assert!(actual_duration < orbital_period);
}
