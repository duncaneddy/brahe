//! Sensitivity matrix computation example.
//!
//! Demonstrates using NumericalSensitivity and AnalyticSensitivity providers
//! to compute ∂f/∂p (sensitivity of dynamics with respect to consider parameters).

use brahe::constants::{GM_EARTH, R_EARTH};
use brahe::math::sensitivity::{
    DAnalyticSensitivity, DNumericalSensitivity, DSensitivityProvider,
};
use brahe::state_koe_to_eci;
use brahe::AngleFormat;
use nalgebra::{DMatrix, DVector, SVector};

fn main() {
    // Dynamics function that takes consider parameters
    let dynamics = |_t: f64, state: &DVector<f64>, params: &DVector<f64>| -> DVector<f64> {
        let cd_area_m = params[0];

        let r = state.fixed_rows::<3>(0);
        let v = state.fixed_rows::<3>(3);
        let r_norm = r.norm();

        // Gravitational acceleration
        let a_grav = -GM_EARTH / (r_norm * r_norm * r_norm) * r;

        // Atmospheric drag (simplified exponential model)
        let h = r_norm - R_EARTH;
        let rho0 = 1.225; // kg/m^3
        let scale_height = 8500.0; // m
        let rho = rho0 * (-h / scale_height).exp();

        let v_norm = v.norm();
        let a_drag = if v_norm > 0.0 {
            -0.5 * rho * cd_area_m * v_norm * v
        } else {
            nalgebra::Vector3::zeros()
        };

        let mut state_dot = DVector::<f64>::zeros(6);
        state_dot
            .fixed_rows_mut::<3>(0)
            .copy_from(&v.clone_owned());
        state_dot
            .fixed_rows_mut::<3>(3)
            .copy_from(&(a_grav + a_drag));
        state_dot
    };

    // Analytical sensitivity function
    let analytical_sensitivity =
        |_t: f64, state: &DVector<f64>, _params: &DVector<f64>| -> DMatrix<f64> {
            let r = state.fixed_rows::<3>(0);
            let v = state.fixed_rows::<3>(3);
            let r_norm = r.norm();

            // Atmospheric density
            let h = r_norm - R_EARTH;
            let rho0 = 1.225;
            let scale_height = 8500.0;
            let rho = rho0 * (-h / scale_height).exp();

            let v_norm = v.norm();

            // ∂(state_dot)/∂(cd_area_m)
            let mut sens = DMatrix::<f64>::zeros(6, 1);
            if v_norm > 0.0 {
                // ∂(a_drag)/∂(cd_area_m) = -0.5 * rho * v_norm * v
                for i in 0..3 {
                    sens[(i + 3, 0)] = -0.5 * rho * v_norm * v[i];
                }
            }

            sens
        };

    // Initial state (400 km LEO circular orbit)
    let oe = SVector::<f64, 6>::from_row_slice(&[R_EARTH + 250e3, 0.001, 51.6, 0.0, 0.0, 0.0]);
    let state_vec = state_koe_to_eci(oe, AngleFormat::Degrees);
    let state = DVector::from_iterator(6, state_vec.iter().copied());

    // Consider parameters
    let params = DVector::from_vec(vec![0.044]); // cd_area_m = Cd*A/m = 2.2*10/500

    // Create numerical sensitivity provider
    let numerical_sens = DNumericalSensitivity::central(Box::new(dynamics.clone()));

    // Compute sensitivity matrix numerically
    let sens_numerical = numerical_sens.compute(0.0, &state, &params);

    println!("Numerical sensitivity (∂f/∂p):");
    println!(
        "  Position rates: [{}, {}, {}]",
        sens_numerical[(0, 0)],
        sens_numerical[(1, 0)],
        sens_numerical[(2, 0)]
    );
    println!(
        "  Velocity rates: [{}, {}, {}]",
        sens_numerical[(3, 0)],
        sens_numerical[(4, 0)],
        sens_numerical[(5, 0)]
    );

    // Create analytical sensitivity provider
    let analytic_sens = DAnalyticSensitivity::new(Box::new(analytical_sensitivity));

    // Compute sensitivity matrix analytically
    let sens_analytical = analytic_sens.compute(0.0, &state, &params);

    println!("\nAnalytical sensitivity (∂f/∂p):");
    println!(
        "  Position rates: [{}, {}, {}]",
        sens_analytical[(0, 0)],
        sens_analytical[(1, 0)],
        sens_analytical[(2, 0)]
    );
    println!(
        "  Velocity rates: [{}, {}, {}]",
        sens_analytical[(3, 0)],
        sens_analytical[(4, 0)],
        sens_analytical[(5, 0)]
    );

    // Compare numerical and analytical
    let diff = &sens_numerical - &sens_analytical;
    let max_diff = diff.abs().max();
    println!("\nMax difference: {:.3e}", max_diff);
}

// Numerical sensitivity (∂f/∂p):
//   Position rates: [0, 0, 0]
//   Velocity rates: [0, -0.000008425648220011794, -0.000010630522385923769]

// Analytical sensitivity (∂f/∂p):
//   Position rates: [0, 0, 0]
//   Velocity rates: [0, -0.000008425648218908737, -0.00001063052238539645]

// Max difference: 1.103e-15