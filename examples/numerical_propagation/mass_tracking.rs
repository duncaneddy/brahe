//! Mass tracking during thrust maneuvers using NumericalOrbitPropagator.
//! Demonstrates using additional_dynamics for mass state and control_input for thrust.

use brahe as bh;
use bh::integrators::traits::DStateDynamics;
use bh::propagators::{DNumericalOrbitPropagator, ForceModelConfiguration, NumericalPropagationConfig};
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Initial orbital elements and state
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0);
    let orbital_state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Extended state: [x, y, z, vx, vy, vz, mass]
    let initial_mass = 1000.0; // kg
    let mut initial_state = na::DVector::zeros(7);
    for i in 0..6 {
        initial_state[i] = orbital_state[i];
    }
    initial_state[6] = initial_mass;

    // Thruster parameters
    let thrust_force = 10.0; // N
    let specific_impulse = 300.0; // s
    let g0 = 9.80665; // m/s^2
    let mass_flow_rate = thrust_force / (specific_impulse * g0); // kg/s

    // Burn duration
    let burn_duration = 600.0; // 10 minutes

    println!("Thruster parameters:");
    println!("  Thrust: {} N", thrust_force);
    println!("  Isp: {} s", specific_impulse);
    println!("  Mass flow rate: {:.2} g/s", mass_flow_rate * 1000.0);
    println!("  Burn duration: {} s", burn_duration);
    println!(
        "  Expected fuel consumption: {:.2} kg",
        mass_flow_rate * burn_duration
    );

    // Additional dynamics for mass tracking
    // Returns full state-sized vector with mass derivative
    let additional_dynamics: DStateDynamics = Box::new(move |t, state, _params| {
        let mut dx = na::DVector::zeros(state.len());
        if t < burn_duration {
            dx[6] = -mass_flow_rate; // dm/dt = -F/(Isp*g0)
        }
        dx
    });

    // Control input for thrust acceleration
    // Returns full state-sized vector with acceleration in indices 3-5
    let control_input = Some(Box::new(move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| {
        let mut dx = na::DVector::zeros(state.len());
        if t < burn_duration {
            let mass = state[6];
            let vel = state.fixed_rows::<3>(3);
            let v_hat = vel.normalize();
            let acc = (thrust_force / mass) * v_hat;
            dx[3] = acc[0];
            dx[4] = acc[1];
            dx[5] = acc[2];
        }
        dx
    }) as Box<dyn Fn(f64, &na::DVector<f64>, Option<&na::DVector<f64>>) -> na::DVector<f64> + Send + Sync>);

    // Create propagator with two-body dynamics (no drag/SRP for clean mass tracking)
    let mut prop = DNumericalOrbitPropagator::new(
        epoch,
        initial_state.clone(),
        NumericalPropagationConfig::default(),
        ForceModelConfiguration::earth_gravity(),
        None, // params
        Some(additional_dynamics),
        control_input,
        None, // initial_covariance
    )
    .unwrap();

    println!("\nInitial state:");
    println!("  Mass: {:.1} kg", initial_mass);
    println!("  Semi-major axis: {:.1} km", oe[0] / 1e3);

    // Propagate through burn and coast
    let total_time = burn_duration + 600.0; // Burn + 10 min coast
    prop.propagate_to(epoch + total_time);

    // Check final state
    let final_state = prop.current_state();
    let final_mass = final_state[6];
    let fuel_consumed = initial_mass - final_mass;

    // Compute final orbital elements
    let final_orbital_state =
        na::SVector::<f64, 6>::from_column_slice(&final_state.as_slice()[..6]);
    let final_koe = bh::state_eci_to_koe(final_orbital_state, bh::AngleFormat::Degrees);

    println!("\nFinal state:");
    println!("  Mass: {:.1} kg", final_mass);
    println!("  Fuel consumed: {:.2} kg", fuel_consumed);
    println!("  Semi-major axis: {:.1} km", final_koe[0] / 1e3);
    println!("  Delta-a: {:.1} km", (final_koe[0] - oe[0]) / 1e3);

    // Verify Tsiolkovsky equation
    let delta_v_expected = specific_impulse * g0 * (initial_mass / final_mass).ln();
    println!("\nTsiolkovsky verification:");
    println!("  Expected delta-v: {:.1} m/s", delta_v_expected);

    // Validate
    let expected_fuel = mass_flow_rate * burn_duration;
    assert!((fuel_consumed - expected_fuel).abs() < 0.1); // Within 0.1 kg
    assert!(final_mass < initial_mass);
    assert!(final_koe[0] > oe[0]); // Orbit raised

    println!("\nExample validated successfully!");
}
