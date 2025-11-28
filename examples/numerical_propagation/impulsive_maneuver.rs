//! Hohmann transfer using impulsive maneuvers with event callbacks.
//! Demonstrates a two-burn orbit transfer from LEO to higher orbit.

use brahe as bh;
use bh::events::{DEventCallback, DTimeEvent, EventAction};
use bh::traits::{DOrbitStateProvider, DStatePropagator};
use nalgebra as na;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Initial circular orbit at 400 km
    let r1 = bh::R_EARTH + 400e3;
    // Target circular orbit at 800 km
    let r2 = bh::R_EARTH + 800e3;

    // Initial state (circular orbit at perigee of transfer)
    let oe_initial = na::SVector::<f64, 6>::new(r1, 0.0001, 0.0, 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe_initial, bh::AngleFormat::Degrees);

    // Calculate Hohmann transfer delta-vs
    let v1_circular = (bh::GM_EARTH / r1).sqrt();
    let v2_circular = (bh::GM_EARTH / r2).sqrt();

    // Transfer ellipse parameters
    let a_transfer = (r1 + r2) / 2.0;
    let v_perigee_transfer = (bh::GM_EARTH * (2.0 / r1 - 1.0 / a_transfer)).sqrt();
    let v_apogee_transfer = (bh::GM_EARTH * (2.0 / r2 - 1.0 / a_transfer)).sqrt();

    // Delta-v magnitudes
    let dv1 = v_perigee_transfer - v1_circular; // First burn (prograde at perigee)
    let dv2 = v2_circular - v_apogee_transfer; // Second burn (prograde at apogee)

    println!(
        "Hohmann Transfer: {:.0} km -> {:.0} km",
        (r1 - bh::R_EARTH) / 1e3,
        (r2 - bh::R_EARTH) / 1e3
    );
    println!("  First burn (perigee):  {:.3} m/s", dv1);
    println!("  Second burn (apogee):  {:.3} m/s", dv2);
    println!("  Total delta-v:         {:.3} m/s", dv1 + dv2);

    // Transfer time (half period of transfer ellipse)
    let transfer_time = PI * (a_transfer.powi(3) / bh::GM_EARTH).sqrt();
    println!("  Transfer time:         {:.1} min", transfer_time / 60.0);

    // Create propagator (two-body for clean Hohmann)
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

    // Track if burns have been applied
    let burn1_applied = Arc::new(AtomicBool::new(false));
    let burn2_applied = Arc::new(AtomicBool::new(false));

    // First burn callback
    let burn1_flag = burn1_applied.clone();
    let first_burn_callback: DEventCallback = Box::new(
        move |_t: bh::Epoch,
              state: &na::DVector<f64>,
              _params: Option<&na::DVector<f64>>|
              -> (
            Option<na::DVector<f64>>,
            Option<na::DVector<f64>>,
            EventAction,
        ) {
            let mut new_state = state.clone();
            // Add delta-v in velocity direction (prograde)
            let v = state.fixed_rows::<3>(3);
            let v_hat = v.normalize();
            new_state[3] += dv1 * v_hat[0];
            new_state[4] += dv1 * v_hat[1];
            new_state[5] += dv1 * v_hat[2];
            burn1_flag.store(true, Ordering::SeqCst);
            println!("  First burn applied at t+0s: dv = {:.3} m/s", dv1);
            (Some(new_state), None, EventAction::Continue)
        },
    );

    // Second burn callback
    let burn2_flag = burn2_applied.clone();
    let epoch_ref = epoch;
    let second_burn_callback: DEventCallback = Box::new(
        move |t: bh::Epoch,
              state: &na::DVector<f64>,
              _params: Option<&na::DVector<f64>>|
              -> (
            Option<na::DVector<f64>>,
            Option<na::DVector<f64>>,
            EventAction,
        ) {
            let mut new_state = state.clone();
            let v = state.fixed_rows::<3>(3);
            let v_hat = v.normalize();
            new_state[3] += dv2 * v_hat[0];
            new_state[4] += dv2 * v_hat[1];
            new_state[5] += dv2 * v_hat[2];
            burn2_flag.store(true, Ordering::SeqCst);
            let dt = t - epoch_ref;
            println!("  Second burn applied at t+{:.1}s: dv = {:.3} m/s", dt, dv2);
            (Some(new_state), None, EventAction::Continue)
        },
    );

    // First burn at t=1s (near immediate)
    let event1 = DTimeEvent::new(epoch + 1.0, "First Burn".to_string())
        .with_callback(first_burn_callback);

    // Second burn at apogee (half transfer period)
    let event2 = DTimeEvent::new(epoch + transfer_time, "Second Burn".to_string())
        .with_callback(second_burn_callback);

    prop.add_event_detector(Box::new(event1));
    prop.add_event_detector(Box::new(event2));

    // Propagate through both burns plus one orbit of final orbit
    let final_orbit_period = 2.0 * PI * (r2.powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + transfer_time + final_orbit_period);

    // Check final orbit
    let final_koe = prop
        .state_koe(prop.current_epoch(), bh::AngleFormat::Degrees)
        .unwrap();
    let final_altitude = final_koe[0] - bh::R_EARTH;

    println!("\nFinal orbit:");
    println!("  Semi-major axis: {:.3} km", final_koe[0] / 1e3);
    println!(
        "  Altitude:        {:.3} km (target: {:.0} km)",
        final_altitude / 1e3,
        (r2 - bh::R_EARTH) / 1e3
    );
    println!("  Eccentricity:    {:.6}", final_koe[1]);

    // Validate final orbit achieved significant altitude gain
    // Note: Some error expected due to numerical integration and event timing
    let altitude_gain = final_altitude - (r1 - bh::R_EARTH);
    assert!(altitude_gain > 200e3); // Significant altitude gain achieved
    assert!(final_koe[1] < 0.1); // Reasonably circular

    println!("\nExample validated successfully!");
}
