//! Configuring numerical propagation integrators.
//! Demonstrates different integrators and tolerance settings.

use brahe as bh;
use bh::integrators::IntegratorConfig;
use bh::propagators::VariationalConfig;
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
        0.001,
        97.8,
        15.0,
        30.0,
        45.0,
    );
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Default configuration: DP54 integrator with standard tolerances
    let config_default = bh::NumericalPropagationConfig::default();
    println!("Default config:");
    println!("  Method: DP54 (Dormand-Prince 5(4))");

    // Custom tolerances using constructor
    let config_tight = bh::NumericalPropagationConfig::new(
        bh::propagators::IntegratorMethod::DP54,
        IntegratorConfig::adaptive(1e-12, 1e-10),
        VariationalConfig::default(),
    );
    println!("\nTight tolerance config:");
    println!("  abs_tol: 1e-12");
    println!("  rel_tol: 1e-10");

    // Create two propagators with different configs
    let mut prop_default = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        config_default,
        bh::ForceModelConfig::two_body_gravity(), // Two-body for cleaner comparison
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let mut prop_tight = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        config_tight,
        bh::ForceModelConfig::two_body_gravity(),
        None,
        None,
        None,
        None,
    )
    .unwrap();

    // Propagate both for 1 orbit
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop_default.propagate_to(epoch + orbital_period);
    prop_tight.propagate_to(epoch + orbital_period);

    // Compare final states
    let final_default = prop_default.current_state();
    let final_tight = prop_tight.current_state();

    let diff = (final_default.fixed_rows::<3>(0) - final_tight.fixed_rows::<3>(0)).norm();
    println!("\nPosition difference between configs: {:.6} m", diff);

    // Validate
    assert!(diff < 0.1); // Configs should give similar results

    println!("\nExample validated successfully!");
}
