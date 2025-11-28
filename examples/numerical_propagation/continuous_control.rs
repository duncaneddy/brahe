//! Propagation with continuous control input (low-thrust).
//! Demonstrates adding continuous thrust acceleration during propagation.

use brahe as bh;
use bh::traits::{DStatePropagator, DOrbitStateProvider};
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Initial circular orbit at 400 km
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 400e3, 0.0001, 0.0, 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Spacecraft parameters
    let mass = 500.0; // kg
    let thrust = 0.1; // N (100 mN thruster - typical ion engine)

    // Define continuous control input: constant tangential thrust
    // Control input must return a derivative vector with the same dimension as state.
    // For 6D orbital state:
    // - Elements 0-2: position derivatives (zeros for control)
    // - Elements 3-5: velocity derivatives (acceleration)
    let control_fn: bh::DControlInput = Some(Box::new(
        move |_t: f64, state_vec: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| {
            let v = state_vec.fixed_rows::<3>(3);
            let v_mag = v.norm();

            // Return full state derivative (same dimension as state)
            let mut dx = na::DVector::zeros(state_vec.len());

            if v_mag > 1e-10 {
                // Unit vector in velocity direction
                let v_hat = v / v_mag;

                // Acceleration from thrust (F = ma -> a = F/m)
                let accel_mag = thrust / mass;
                dx[3] = accel_mag * v_hat[0];
                dx[4] = accel_mag * v_hat[1];
                dx[5] = accel_mag * v_hat[2];
            }

            dx
        },
    ));

    // Create propagator with continuous control
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfig::two_body_gravity(), // Two-body + control
        None,
        None,
        control_fn,
        None,
    )
    .unwrap();

    // Also create reference propagator without thrust
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

    // Propagate for 10 orbits
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    let end_time = epoch + 10.0 * orbital_period;

    prop.propagate_to(end_time);
    prop_ref.propagate_to(end_time);

    // Compare orbits
    let koe_thrust = prop
        .state_koe(end_time, bh::AngleFormat::Degrees)
        .unwrap();
    let koe_ref = prop_ref
        .state_koe(end_time, bh::AngleFormat::Degrees)
        .unwrap();

    let alt_thrust = koe_thrust[0] - bh::R_EARTH;
    let alt_ref = koe_ref[0] - bh::R_EARTH;
    let alt_gain = alt_thrust - alt_ref;

    // Calculate total delta-v applied
    let total_time = 10.0 * orbital_period;
    let dv_total = (thrust / mass) * total_time;

    println!("Low-Thrust Orbit Raising (10 orbits):");
    println!("  Thrust: {:.1} mN", thrust * 1000.0);
    println!("  Spacecraft mass: {:.0} kg", mass);
    println!("  Acceleration: {:.2} micro-m/s^2", thrust / mass * 1e6);
    println!("\nAfter {:.1} hours:", 10.0 * orbital_period / 3600.0);
    println!("  Reference altitude: {:.3} km", alt_ref / 1e3);
    println!("  With thrust altitude: {:.3} km", alt_thrust / 1e3);
    println!("  Altitude gain: {:.3} km", alt_gain / 1e3);
    println!("  Total delta-v applied: {:.3} m/s", dv_total);

    // Validate - thrust should raise orbit
    assert!(alt_thrust > alt_ref);
    assert!(alt_gain > 0.0);

    println!("\nExample validated successfully!");
}
