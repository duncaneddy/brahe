//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.34"
//! ```

//! Sensitivity matrix propagation pattern example.
//!
//! Demonstrates propagating the sensitivity matrix alongside state to map
//! parameter uncertainties to state uncertainties over time.
//!
//! The sensitivity matrix Φ = ∂x/∂p evolves according to:
//!     dΦ/dt = (∂f/∂x) * Φ + (∂f/∂p)
//!
//! This augmented state approach propagates [state, vec(Φ)] together.

use brahe::constants::{GM_EARTH, R_EARTH};
use brahe::integrators::*;
use brahe::math::jacobian::{DJacobianProvider, DNumericalJacobian};
use brahe::math::sensitivity::{DNumericalSensitivity, DSensitivityProvider};
use brahe::{state_osculating_to_cartesian, AngleFormat};
use nalgebra::{DVector, SVector};

fn main() {
    // Consider parameters
    let cd_area_m = 2.2 * 10.0 / 500.0; // Cd=2.2, A=10m^2, m=500kg
    let params = DVector::from_vec(vec![cd_area_m]);
    let num_params = params.len();

    // Dynamics function with parameters
    let dynamics_with_params =
        |_t: f64, state: &DVector<f64>, params: &DVector<f64>| -> DVector<f64> {
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

    // Create sensitivity provider
    let sensitivity_provider =
        DNumericalSensitivity::central(Box::new(dynamics_with_params.clone()));

    // Create Jacobian provider (dynamics without explicit params for Jacobian)
    let params_clone = params.clone();
    let dynamics_for_jacobian = move |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        let cd_area_m = params_clone[0];

        let r = state.fixed_rows::<3>(0);
        let v = state.fixed_rows::<3>(3);
        let r_norm = r.norm();

        let a_grav = -GM_EARTH / (r_norm * r_norm * r_norm) * r;

        let h = r_norm - R_EARTH;
        let rho0 = 1.225;
        let scale_height = 8500.0;
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

    let jacobian_provider = DNumericalJacobian::central(Box::new(dynamics_for_jacobian));

    // Augmented dynamics for state + sensitivity matrix propagation
    let params_for_aug = params.clone();
    let augmented_dynamics = move |t: f64,
                                   aug_state: DVector<f64>,
                                   _params: Option<&DVector<f64>>|
          -> DVector<f64> {
        // Extract state and sensitivity matrix
        let state = aug_state.rows(0, 6).into_owned();
        let phi_flat = aug_state.rows(6, 6 * num_params).into_owned();

        // Reshape phi from flat vector to matrix (column-major)
        let mut phi = nalgebra::DMatrix::<f64>::zeros(6, num_params);
        for j in 0..num_params {
            for i in 0..6 {
                phi[(i, j)] = phi_flat[j * 6 + i];
            }
        }

        // State derivative
        let state_dot = dynamics_with_params(t, &state, &params_for_aug);

        // Compute Jacobian ∂f/∂x
        let jacobian = jacobian_provider.compute(t, state.clone(), None);

        // Compute sensitivity ∂f/∂p
        let sensitivity = sensitivity_provider.compute(t, &state, &params_for_aug);

        // Sensitivity matrix derivative: dΦ/dt = J*Φ + S
        let phi_dot = &jacobian * &phi + &sensitivity;

        // Flatten phi_dot back to vector (column-major)
        let mut phi_dot_flat = DVector::<f64>::zeros(6 * num_params);
        for j in 0..num_params {
            for i in 0..6 {
                phi_dot_flat[j * 6 + i] = phi_dot[(i, j)];
            }
        }

        // Concatenate state_dot and phi_dot
        let mut aug_dot = DVector::<f64>::zeros(6 + 6 * num_params);
        aug_dot.rows_mut(0, 6).copy_from(&state_dot);
        aug_dot.rows_mut(6, 6 * num_params).copy_from(&phi_dot_flat);
        aug_dot
    };

    // Initial state (200 km LEO for significant drag effects)
    let oe = SVector::<f64, 6>::from_row_slice(&[R_EARTH + 200e3, 0.001, 51.6, 0.0, 0.0, 0.0]);
    let state_vec = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
    let state = DVector::from_iterator(6, state_vec.iter().copied());

    // Initial sensitivity matrix (zeros - we're interested in how it develops)
    let phi0 = DVector::<f64>::zeros(6 * num_params);

    // Augmented initial state
    let mut aug_state = DVector::<f64>::zeros(6 + 6 * num_params);
    aug_state.rows_mut(0, 6).copy_from(&state);
    aug_state.rows_mut(6, 6 * num_params).copy_from(&phi0);

    // Create integrator for augmented system
    // Using fixed step RK4 for simplicity and exact parity with Python
    let aug_dim = 6 + 6 * num_params;
    let config = IntegratorConfig::fixed_step(1.0);
    let integrator = RK4DIntegrator::with_config(
        aug_dim,
        Box::new(augmented_dynamics),
        None,
        None,
        None,
        config,
    );

    // Propagate for 1 hour
    let t_final = 3600.0;
    let mut t = 0.0;
    let dt = 1.0;

    while t < t_final {
        aug_state = integrator.step(t, aug_state, None).state;
        t += dt;
    }

    // Extract final state and sensitivity matrix
    let final_state = aug_state.rows(0, 6).into_owned();
    let final_phi_flat = aug_state.rows(6, 6 * num_params).into_owned();

    // Reshape to matrix
    let mut final_phi = nalgebra::DMatrix::<f64>::zeros(6, num_params);
    for j in 0..num_params {
        for i in 0..6 {
            final_phi[(i, j)] = final_phi_flat[j * 6 + i];
        }
    }

    println!("Final position after {} minutes:", t_final / 60.0);
    println!("  x: {:.3} km", final_state[0] / 1000.0);
    println!("  y: {:.3} km", final_state[1] / 1000.0);
    println!("  z: {:.3} km", final_state[2] / 1000.0);

    println!("\nSensitivity matrix Φ = ∂x/∂p (position per unit Cd*A/m):");
    println!("  dx/dp: {:.3} m/(m²/kg)", final_phi[(0, 0)]);
    println!("  dy/dp: {:.3} m/(m²/kg)", final_phi[(1, 0)]);
    println!("  dz/dp: {:.3} m/(m²/kg)", final_phi[(2, 0)]);

    println!("\nSensitivity matrix Φ = ∂x/∂p (velocity per unit Cd*A/m):");
    println!("  dvx/dp: {:.6} m/s/(m²/kg)", final_phi[(3, 0)]);
    println!("  dvy/dp: {:.6} m/s/(m²/kg)", final_phi[(4, 0)]);
    println!("  dvz/dp: {:.6} m/s/(m²/kg)", final_phi[(5, 0)]);

    // Interpret: If we have 10% uncertainty in Cd*A/m,
    // the position uncertainty after 1 hour would be:
    let delta_p = 0.1 * cd_area_m;
    let pos_uncertainty = (final_phi[(0, 0)].powi(2)
        + final_phi[(1, 0)].powi(2)
        + final_phi[(2, 0)].powi(2))
    .sqrt()
        * delta_p;
    println!(
        "\nPosition uncertainty for 10% parameter uncertainty: {:.1} m",
        pos_uncertainty
    );
}

// Example output:
// Final position after 60 minutes:
//   x: -2884.245 km
//   y: -3673.659 km
//   z: -4635.004 km

// Sensitivity matrix Φ = ∂x/∂p (position per unit Cd*A/m):
//   dx/dp: 59942.895 m/(m²/kg)
//   dy/dp: -3796.877 m/(m²/kg)
//   dz/dp: -4790.467 m/(m²/kg)

// Sensitivity matrix Φ = ∂x/∂p (velocity per unit Cd*A/m):
//   dvx/dp: 44.091415 m/s/(m²/kg)
//   dvy/dp: 33.444232 m/s/(m²/kg)
//   dvz/dp: 42.196119 m/s/(m²/kg)

// Position uncertainty for 10% parameter uncertainty: 265.1 m
