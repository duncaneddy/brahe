//! Covariance propagation using the State Transition Matrix.
//! Demonstrates propagating initial uncertainty through orbital dynamics.

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

    // Create propagation config with STM enabled
    let mut prop_config = bh::NumericalPropagationConfig::default();
    prop_config.variational.enable_stm = true;

    // Create propagator (two-body for clean demonstration)
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

    // Define initial covariance (diagonal)
    // Position uncertainty: 10 m in each axis (100 m² variance)
    // Velocity uncertainty: 0.01 m/s in each axis (0.0001 m²/s² variance)
    let p0: na::DMatrix<f64> = na::DMatrix::from_diagonal(&na::DVector::from_vec(vec![
        100.0, 100.0, 100.0, 0.0001, 0.0001, 0.0001,
    ]));

    println!("Initial Covariance (diagonal, sqrt):");
    println!("  Position std: {:.1} m", p0[(0, 0)].sqrt());
    println!("  Velocity std: {:.2} mm/s", p0[(3, 3)].sqrt() * 1000.0);

    // Propagate for one orbital period
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + orbital_period);

    // Get the State Transition Matrix
    let stm = prop.stm().expect("STM should be available").clone();
    println!("\nSTM shape: ({}, {})", stm.nrows(), stm.ncols());

    // Propagate covariance: P(t) = Phi @ P0 @ Phi^T
    let p = &stm * &p0 * stm.transpose();

    // Extract position and velocity uncertainties
    let pos_std_x = p[(0, 0)].sqrt();
    let pos_std_y = p[(1, 1)].sqrt();
    let pos_std_z = p[(2, 2)].sqrt();
    let vel_std_x = p[(3, 3)].sqrt() * 1000.0;
    let vel_std_y = p[(4, 4)].sqrt() * 1000.0;
    let vel_std_z = p[(5, 5)].sqrt() * 1000.0;

    println!("\nPropagated Covariance after one orbit:");
    println!(
        "  Position std (x,y,z): ({:.1}, {:.1}, {:.1}) m",
        pos_std_x, pos_std_y, pos_std_z
    );
    println!(
        "  Velocity std (x,y,z): ({:.2}, {:.2}, {:.2}) mm/s",
        vel_std_x, vel_std_y, vel_std_z
    );

    // Compute position uncertainty magnitude
    let pos_uncertainty_initial = (p0[(0, 0)] + p0[(1, 1)] + p0[(2, 2)]).sqrt();
    let pos_uncertainty_final = (p[(0, 0)] + p[(1, 1)] + p[(2, 2)]).sqrt();

    println!("\nTotal position uncertainty:");
    println!("  Initial: {:.1} m", pos_uncertainty_initial);
    println!("  Final:   {:.1} m", pos_uncertainty_final);
    println!(
        "  Growth:  {:.1}x",
        pos_uncertainty_final / pos_uncertainty_initial
    );

    // Validate that covariance was propagated
    assert_eq!(stm.nrows(), 6);
    assert_eq!(stm.ncols(), 6);
    assert!(pos_uncertainty_final >= pos_uncertainty_initial); // Uncertainty grows

    println!("\nExample validated successfully!");
}
