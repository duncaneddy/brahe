//! Variable thrust control with ramp-up and ramp-down phases.
//! Demonstrates time-varying thrust profiles during propagation.

use brahe as bh;
use bh::traits::{DOrbitStateProvider, DStatePropagator};
use nalgebra as na;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Initial circular orbit at 400 km
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 400e3, 0.0001, 0.0, 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Spacecraft and maneuver parameters
    let mass = 500.0; // kg
    let max_thrust = 0.5; // N (500 mN thruster)
    let ramp_time = 300.0; // s (5 minute ramp)
    let burn_duration = 1800.0; // s (30 minute burn)
    let maneuver_start_offset = 600.0; // Start 10 minutes into propagation

    // Define variable thrust control input
    // The closure captures the maneuver parameters
    let control_fn: bh::DControlInput = Some(Box::new(
        move |t: f64, state_vec: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| {
            // Return zeros if outside burn window
            let mut dx = na::DVector::zeros(state_vec.len());

            // Time since maneuver start (t is seconds since epoch)
            let t_maneuver = t - maneuver_start_offset;

            // Check if within burn window
            if t_maneuver < 0.0 || t_maneuver > burn_duration {
                return dx;
            }

            // Compute thrust magnitude with ramp profile
            let magnitude = if t_maneuver < ramp_time {
                // Ramp up phase
                max_thrust * (t_maneuver / ramp_time)
            } else if t_maneuver > burn_duration - ramp_time {
                // Ramp down phase
                max_thrust * ((burn_duration - t_maneuver) / ramp_time)
            } else {
                // Constant thrust phase
                max_thrust
            };

            // Thrust direction along velocity
            let v = state_vec.fixed_rows::<3>(3);
            let v_mag = v.norm();

            if v_mag > 1e-10 {
                let v_hat = v / v_mag;
                // Acceleration from thrust (F = ma -> a = F/m)
                let accel_mag = magnitude / mass;
                dx[3] = accel_mag * v_hat[0];
                dx[4] = accel_mag * v_hat[1];
                dx[5] = accel_mag * v_hat[2];
            }

            dx
        },
    ));

    // Create propagator with variable thrust control
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfig::two_body_gravity(),
        None,
        None,
        control_fn,
        None,
    )
    .unwrap();

    // Create reference propagator without thrust
    let mut prop_ref = bh::DNumericalOrbitPropagator::new(
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

    // Propagate for duration covering the entire maneuver
    let end_time = epoch + 3600.0; // 1 hour (covers 30-min burn starting at 10 min)

    prop.propagate_to(end_time);
    prop_ref.propagate_to(end_time);

    // Compare final orbits
    let koe_thrust = prop.state_koe(end_time, bh::AngleFormat::Degrees).unwrap();
    let koe_ref = prop_ref
        .state_koe(end_time, bh::AngleFormat::Degrees)
        .unwrap();

    let alt_thrust = koe_thrust[0] - bh::R_EARTH;
    let alt_ref = koe_ref[0] - bh::R_EARTH;
    let alt_gain = alt_thrust - alt_ref;

    // Calculate approximate delta-v (trapezoidal profile integration)
    // Full thrust duration minus ramp portions: burn_duration - ramp_time
    let effective_time = burn_duration - ramp_time;
    let dv_approx = (max_thrust / mass) * effective_time;

    println!("Variable Thrust Orbit Raising:");
    println!("  Max thrust: {:.1} mN", max_thrust * 1000.0);
    println!("  Spacecraft mass: {:.0} kg", mass);
    println!(
        "  Burn duration: {:.0} s ({:.0} min)",
        burn_duration,
        burn_duration / 60.0
    );
    println!(
        "  Ramp time: {:.0} s ({:.0} min)",
        ramp_time,
        ramp_time / 60.0
    );
    println!("\nAfter 1 hour propagation:");
    println!("  Reference altitude: {:.3} km", alt_ref / 1e3);
    println!("  With thrust altitude: {:.3} km", alt_thrust / 1e3);
    println!("  Altitude gain: {:.3} km", alt_gain / 1e3);
    println!("  Approx delta-v applied: {:.3} m/s", dv_approx);

    // Validate - thrust should raise orbit
    assert!(alt_thrust > alt_ref, "Thrust should raise orbit");
    assert!(alt_gain > 0.0, "Altitude gain should be positive");

    println!("\nExample validated successfully!");
}
