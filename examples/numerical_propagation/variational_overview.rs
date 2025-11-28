//! Overview of variational propagation: STM, covariance, and sensitivity.
//! Demonstrates enabling and using all variational features together.

use brahe as bh;
use bh::traits::{DOrbitCovarianceProvider, DStatePropagator};
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    // Create initial epoch and state (LEO satellite)
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Define spacecraft parameters: [mass, drag_area, Cd, srp_area, Cr]
    let params = na::DVector::from_vec(vec![500.0, 2.0, 2.2, 2.0, 1.3]);

    // Create propagation config enabling STM and sensitivity with history storage
    let mut prop_config = bh::NumericalPropagationConfig::default();
    prop_config.variational.enable_stm = true;
    prop_config.variational.store_stm_history = true;
    prop_config.variational.enable_sensitivity = true;
    prop_config.variational.store_sensitivity_history = true;

    // Define initial covariance (diagonal)
    // Position uncertainty: 10 m (variance = 100 m²)
    // Velocity uncertainty: 0.01 m/s (variance = 0.0001 m²/s²)
    let p0: na::DMatrix<f64> = na::DMatrix::from_diagonal(&na::DVector::from_vec(vec![
        100.0, 100.0, 100.0, 0.0001, 0.0001, 0.0001,
    ]));

    // Create propagator with full force model
    // Arguments: epoch, state, config, force_config, params, additional_dynamics, control_input, initial_covariance
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        prop_config,
        bh::ForceModelConfig::default(),
        Some(params.clone()),
        None,  // additional_dynamics
        None,  // control_input
        Some(p0.clone()),  // initial_covariance
    )
    .unwrap();

    println!("=== Variational Propagation Overview ===\n");
    println!("Initial State:");
    println!("  Semi-major axis: {:.1} km", oe[0] / 1000.0);
    println!("  Position std: {:.1} m", p0[(0, 0)].sqrt());
    println!("  Velocity std: {:.2} mm/s", p0[(3, 3)].sqrt() * 1000.0);

    // Propagate for one orbital period
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + orbital_period);

    // === STM Access ===
    println!("\n--- State Transition Matrix (STM) ---");
    let stm = prop.stm().expect("STM should be available");
    println!("STM shape: ({}, {})", stm.nrows(), stm.ncols());
    println!(
        "STM determinant: {:.6} (should be ~1 for conservative forces)",
        stm.determinant()
    );

    // STM at intermediate time (half orbit)
    let stm_half = prop.stm_at(epoch + orbital_period / 2.0);
    println!("STM at t/2 available: {}", stm_half.is_some());

    // === Covariance Propagation ===
    println!("\n--- Covariance Propagation ---");

    // Manual propagation: P(t) = STM @ P0 @ STM^T
    let _p_manual = stm * &p0 * stm.transpose();

    // Using built-in covariance retrieval
    let p_gcrf = prop.covariance_gcrf(epoch + orbital_period).unwrap();
    let p_rtn = prop.covariance_rtn(epoch + orbital_period).unwrap();

    // Extract position uncertainties
    println!("Position std (GCRF frame):");
    println!(
        "  X: {:.1} m, Y: {:.1} m, Z: {:.1} m",
        p_gcrf[(0, 0)].sqrt(),
        p_gcrf[(1, 1)].sqrt(),
        p_gcrf[(2, 2)].sqrt()
    );
    println!("Position std (RTN frame):");
    println!(
        "  R: {:.1} m, T: {:.1} m, N: {:.1} m",
        p_rtn[(0, 0)].sqrt(),
        p_rtn[(1, 1)].sqrt(),
        p_rtn[(2, 2)].sqrt()
    );

    // === Sensitivity Analysis ===
    println!("\n--- Parameter Sensitivity ---");
    let sens = prop.sensitivity().expect("Sensitivity should be available");
    println!("Sensitivity matrix shape: ({}, {})", sens.nrows(), sens.ncols());

    // Position sensitivity magnitude to each parameter
    let param_names = ["mass", "drag_area", "Cd", "srp_area", "Cr"];
    println!("\nPosition sensitivity to 1% parameter uncertainty:");
    for (i, name) in param_names.iter().enumerate() {
        let pos_sens = na::Vector3::new(sens[(0, i)], sens[(1, i)], sens[(2, i)]);
        let pos_sens_mag = pos_sens.norm();
        let param_uncertainty = params[i] * 0.01; // 1% uncertainty
        let pos_error = pos_sens_mag * param_uncertainty;
        println!("  {:10}: {:.2} m", name, pos_error);
    }

    // === Summary ===
    println!("\n--- Summary ---");
    let total_pos_std_initial = (p0[(0, 0)] + p0[(1, 1)] + p0[(2, 2)]).sqrt();
    let total_pos_std_final = (p_gcrf[(0, 0)] + p_gcrf[(1, 1)] + p_gcrf[(2, 2)]).sqrt();
    println!(
        "Total position uncertainty: {:.1} m -> {:.1} m",
        total_pos_std_initial, total_pos_std_final
    );
    println!(
        "Uncertainty growth factor: {:.1}x",
        total_pos_std_final / total_pos_std_initial
    );

    // Validate outputs
    assert_eq!(stm.nrows(), 6);
    assert_eq!(stm.ncols(), 6);
    assert_eq!(sens.nrows(), 6);
    assert_eq!(sens.ncols(), 5);
    assert_eq!(p_gcrf.nrows(), 6);
    assert_eq!(p_rtn.nrows(), 6);
    assert!(total_pos_std_final >= total_pos_std_initial);

    println!("\nExample validated successfully!");
}
