//! Parameter sensitivity analysis using NumericalOrbitPropagator.
//! Demonstrates computing sensitivity of orbital state to configuration parameters.

use brahe as bh;
use bh::traits::DStatePropagator;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 400e3, 0.01, 45.0, 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Create propagation config with sensitivity enabled
    let mut prop_config = bh::NumericalPropagationConfig::default();
    prop_config.variational.enable_sensitivity = true;
    prop_config.variational.store_sensitivity_history = true;

    // Define spacecraft parameters: [mass, drag_area, Cd, srp_area, Cr]
    let params = na::DVector::from_vec(vec![500.0, 2.0, 2.2, 2.0, 1.3]);

    // Create propagator with full force model (needed for parameter sensitivity)
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        prop_config,
        bh::ForceModelConfig::default(),
        Some(params.clone()),
        None,
        None,
        None,
    )
    .unwrap();

    println!("Spacecraft Parameters:");
    println!("  Mass: {:.1} kg", params[0]);
    println!("  Drag area: {:.1} m²", params[1]);
    println!("  Drag coefficient (Cd): {:.1}", params[2]);
    println!("  SRP area: {:.1} m²", params[3]);
    println!("  SRP coefficient (Cr): {:.1}", params[4]);

    // Propagate for one orbital period
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + orbital_period);

    // Get the sensitivity matrix (6 x 5)
    if let Some(sens) = prop.sensitivity() {
        println!(
            "\nSensitivity Matrix shape: ({}, {})",
            sens.nrows(),
            sens.ncols()
        );
        println!("(Rows: state components [x,y,z,vx,vy,vz], Cols: params [mass,A_d,Cd,A_s,Cr])");

        let param_names = ["mass", "drag_area", "Cd", "srp_area", "Cr"];

        println!("\nPosition sensitivity magnitude to each parameter:");
        for (i, name) in param_names.iter().enumerate() {
            // Position sensitivity magnitude for this parameter
            let pos_sens = na::Vector3::new(sens[(0, i)], sens[(1, i)], sens[(2, i)]);
            let mag = pos_sens.norm();
            println!("  {:10}: {:.3e} m per unit param", name, mag);
        }

        // Compute impact of 1% parameter uncertainties
        println!("\nPosition error from 1% parameter uncertainty:");
        let param_uncertainties: Vec<f64> = params.iter().map(|p| p * 0.01).collect();

        let mut total_pos_error_sq = 0.0;
        for (i, name) in param_names.iter().enumerate() {
            let pos_sens = na::Vector3::new(sens[(0, i)], sens[(1, i)], sens[(2, i)]);
            let pos_error = pos_sens.norm() * param_uncertainties[i];
            total_pos_error_sq += pos_error.powi(2);
            println!("  {:10}: {:.1} m", name, pos_error);
        }

        let total_pos_error = total_pos_error_sq.sqrt();
        println!("\n  Total (RSS): {:.1} m", total_pos_error);

        // Validate
        assert_eq!(sens.nrows(), 6);
        assert_eq!(sens.ncols(), 5);
        println!("\nExample validated successfully!");
    } else {
        println!("\nSensitivity not available (may require full force model)");
    }
}
