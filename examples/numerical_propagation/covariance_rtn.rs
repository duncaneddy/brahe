//! Covariance propagation in RTN (Radial-Tangential-Normal) frame.
//! Demonstrates frame-specific covariance retrieval and physical interpretation.

use brahe as bh;
use bh::traits::{DOrbitCovarianceProvider, DStatePropagator};
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Enable STM for covariance propagation
    let mut prop_config = bh::NumericalPropagationConfig::default();
    prop_config.variational.enable_stm = true;
    prop_config.variational.store_stm_history = true;

    // Define initial covariance in ECI frame
    // Position uncertainty: 10 m in each axis
    // Velocity uncertainty: 0.01 m/s in each axis
    let p0: na::DMatrix<f64> = na::DMatrix::from_diagonal(&na::DVector::from_vec(vec![
        100.0, 100.0, 100.0, 0.0001, 0.0001, 0.0001,
    ]));

    // Create propagator with initial covariance
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        prop_config,
        bh::ForceModelConfig::two_body_gravity(),
        None,
        None,
        None,
        Some(p0),
    )
    .unwrap();

    println!("=== Covariance in RTN Frame ===\n");
    println!("Initial position std (ECI): 10.0 m in each axis");

    // Propagate for one orbital period
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + orbital_period);

    // Get covariance in different frames
    let target = epoch + orbital_period;
    let p_gcrf = prop.covariance_gcrf(target).unwrap();
    let p_rtn = prop.covariance_rtn(target).unwrap();

    println!("\n--- GCRF Frame Results ---");
    println!("Position std (X, Y, Z):");
    println!("  X: {:.1} m", p_gcrf[(0, 0)].sqrt());
    println!("  Y: {:.1} m", p_gcrf[(1, 1)].sqrt());
    println!("  Z: {:.1} m", p_gcrf[(2, 2)].sqrt());

    println!("\n--- RTN Frame Results ---");
    println!("Position std (R, T, N):");
    println!(
        "  Radial (R):     {:.1} m  <- Altitude uncertainty",
        p_rtn[(0, 0)].sqrt()
    );
    println!(
        "  Tangential (T): {:.1} m  <- Along-track timing",
        p_rtn[(1, 1)].sqrt()
    );
    println!(
        "  Normal (N):     {:.1} m  <- Cross-track offset",
        p_rtn[(2, 2)].sqrt()
    );

    // Physical interpretation
    println!("\n--- Physical Interpretation ---");
    println!("RTN frame aligns with the orbit:");
    println!("  R (Radial): Points from Earth center to satellite");
    println!("  T (Tangential): Points along velocity direction");
    println!("  N (Normal): Completes right-hand system (cross-track)");
    println!();
    println!("Key insight: Along-track (T) uncertainty grows fastest because");
    println!("velocity uncertainty causes timing errors that accumulate.");
    println!(
        "After one orbit: T/R ratio = {:.1}x",
        p_rtn[(1, 1)].sqrt() / p_rtn[(0, 0)].sqrt()
    );

    // Show correlation structure
    println!("\n--- Position Correlation Matrix (RTN) ---");
    let std_r = p_rtn[(0, 0)].sqrt();
    let std_t = p_rtn[(1, 1)].sqrt();
    let std_n = p_rtn[(2, 2)].sqrt();

    println!("       R      T      N");
    println!(
        "  R  {:6.3} {:6.3} {:6.3}",
        p_rtn[(0, 0)] / (std_r * std_r),
        p_rtn[(0, 1)] / (std_r * std_t),
        p_rtn[(0, 2)] / (std_r * std_n)
    );
    println!(
        "  T  {:6.3} {:6.3} {:6.3}",
        p_rtn[(1, 0)] / (std_t * std_r),
        p_rtn[(1, 1)] / (std_t * std_t),
        p_rtn[(1, 2)] / (std_t * std_n)
    );
    println!(
        "  N  {:6.3} {:6.3} {:6.3}",
        p_rtn[(2, 0)] / (std_n * std_r),
        p_rtn[(2, 1)] / (std_n * std_t),
        p_rtn[(2, 2)] / (std_n * std_n)
    );

    // Validate
    assert_eq!(p_gcrf.nrows(), 6);
    assert_eq!(p_rtn.nrows(), 6);
    assert!(p_rtn[(1, 1)].sqrt() > p_rtn[(0, 0)].sqrt()); // T > R

    println!("\nExample validated successfully!");
}
