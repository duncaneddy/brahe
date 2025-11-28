//! State Transition Matrix (STM) propagation.
//! Demonstrates enabling STM computation and accessing results.

use brahe as bh;
use bh::traits::DStatePropagator;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Enable STM via config
    let mut prop_config = bh::NumericalPropagationConfig::default();
    prop_config.variational.enable_stm = true;
    prop_config.variational.store_stm_history = true;

    // Create propagator with two-body gravity
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        prop_config,
        bh::ForceModelConfig::two_body_gravity(),
        None,
        None,
        None,
        None,
    )
    .unwrap();

    println!("=== STM Propagation Example ===\n");

    // Propagate for one orbital period
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + orbital_period);

    // Access STM at final time
    let stm = prop.stm().expect("STM should be available");
    println!("Final STM shape: ({}, {})", stm.nrows(), stm.ncols());
    println!("STM determinant: {:.6}", stm.determinant());

    // STM at initial time should be identity
    let stm_initial = prop.stm_at(epoch).expect("STM at t=0 should be available");
    let identity = na::DMatrix::identity(6, 6);
    let max_diff = (&stm_initial - &identity).abs().max();
    println!("\nSTM at t=0 (should be identity):");
    println!("  Max off-diagonal: {:.2e}", max_diff);

    // STM at intermediate time
    let half_period = epoch + orbital_period / 2.0;
    let stm_half = prop.stm_at(half_period).expect("STM at t=T/2 should be available");
    println!("\nSTM at t=T/2:");
    println!("  Determinant: {:.6}", stm_half.determinant());

    // STM composition property verification
    let stm_inv = stm.clone().try_inverse().expect("STM should be invertible");
    let identity_check = stm * &stm_inv;
    let max_deviation = (&identity_check - &identity).abs().max();
    println!("\nSTM * STM^-1 (should be identity):");
    println!("  Max deviation from I: {:.2e}", max_deviation);

    // STM structure interpretation
    println!("\n=== STM Structure ===");
    println!("Upper-left 3x3: Position sensitivity to initial position");
    println!("Upper-right 3x3: Position sensitivity to initial velocity");
    println!("Lower-left 3x3: Velocity sensitivity to initial position");
    println!("Lower-right 3x3: Velocity sensitivity to initial velocity");

    // Show magnitude of each block
    let stm = prop.stm().unwrap();
    let pos_pos = stm.view((0, 0), (3, 3)).norm();
    let pos_vel = stm.view((0, 3), (3, 3)).norm();
    let vel_pos = stm.view((3, 0), (3, 3)).norm();
    let vel_vel = stm.view((3, 3), (3, 3)).norm();

    println!("\nBlock Frobenius norms after one orbit:");
    println!("  dr/dr0: {:.2}", pos_pos);
    println!("  dr/dv0: {:.2}", pos_vel);
    println!("  dv/dr0: {:.6}", vel_pos);
    println!("  dv/dv0: {:.2}", vel_vel);

    // Validate
    assert_eq!(stm.nrows(), 6);
    assert_eq!(stm.ncols(), 6);
    assert!((stm.determinant() - 1.0).abs() < 1e-6); // Hamiltonian system preserves volume

    println!("\nExample validated successfully!");
}
