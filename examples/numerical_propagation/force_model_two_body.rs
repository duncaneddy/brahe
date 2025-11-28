//! Using two-body (point mass) force model.
//! No perturbations - equivalent to Keplerian propagation.

use brahe as bh;
use bh::traits::{DStatePropagator, SStatePropagator};
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,
        0.01,
        45.0,
        0.0,
        0.0,
        0.0,
    );
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Two-body force model - point mass gravity only
    let force_config = bh::ForceModelConfig::two_body_gravity();

    // No parameters required for two-body
    println!("Two-body requires params: {}", force_config.requires_params());

    // Create propagator (no params needed)
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::NumericalPropagationConfig::default(),
        force_config.clone(),
        None, // No params required
        None,
        None,
        None,
    )
    .unwrap();

    // Create Keplerian propagator for comparison (use Cartesian state for direct comparison)
    let mut kep_prop = bh::KeplerianPropagator::from_eci(
        epoch,
        state,
        60.0, // step size
    );

    // Propagate both for 10 orbits
    let sma = bh::R_EARTH + 500e3;
    let orbital_period = 2.0 * PI * (sma.powi(3) / bh::GM_EARTH).sqrt();
    let end_epoch = epoch + 10.0 * orbital_period;

    prop.propagate_to(end_epoch);
    kep_prop.propagate_to(end_epoch);

    // Compare final states
    let num_state = prop.current_state();
    let kep_state = kep_prop.current_state();

    // Convert static Vector6 to dynamic for comparison
    let kep_dvec = na::DVector::from_column_slice(kep_state.as_slice());
    let pos_diff = (num_state.fixed_rows::<3>(0) - kep_dvec.fixed_rows::<3>(0)).norm();
    let vel_diff = (num_state.fixed_rows::<3>(3) - kep_dvec.fixed_rows::<3>(3)).norm();

    println!(
        "\nAfter 10 orbits ({:.1} hours):",
        10.0 * orbital_period / 3600.0
    );
    println!("  Position difference: {:.6} m", pos_diff);
    println!("  Velocity difference: {:.9} m/s", vel_diff);

    // Validate - should be close (numerical integration error accumulates over time)
    assert!(!force_config.requires_params());
    assert!(pos_diff < 50.0); // Less than 50 meter difference over 10 orbits
    assert!(vel_diff < 0.05); // Less than 5 cm/s difference

    println!("\nExample validated successfully!");
}
