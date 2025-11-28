//! Demonstrates equivalence between STM propagation and direct perturbation integration.
//!
//! This example shows that for small perturbations, the State Transition Matrix (STM)
//! can accurately predict the effect of initial perturbations without directly
//! integrating the perturbed trajectory. This is fundamental for orbit determination
//! and covariance propagation.

//! ```cargo
//! [dependencies]
//! brahe = { path = "../../" }
//! nalgebra = "0.33"
//! ```

use brahe::*;
use nalgebra::{DMatrix, DVector, SVector};

/// Two-body point-mass dynamics with Earth gravity (for integrator)
fn dynamics(_t: f64, state: &DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
    let r = state.rows(0, 3);
    let v = state.rows(3, 3);
    let r_norm = r.norm();
    let a = -GM_EARTH / (r_norm * r_norm * r_norm) * r;

    let mut state_dot = DVector::zeros(6);
    state_dot.rows_mut(0, 3).copy_from(&v);
    state_dot.rows_mut(3, 3).copy_from(&a);
    state_dot
}

/// Two-body dynamics (for Jacobian computation - no params)
fn dynamics_for_jac(_t: f64, state: &DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
    let r = state.rows(0, 3);
    let v = state.rows(3, 3);
    let r_norm = r.norm();
    let a = -GM_EARTH / (r_norm * r_norm * r_norm) * r;

    let mut state_dot = DVector::zeros(6);
    state_dot.rows_mut(0, 3).copy_from(&v);
    state_dot.rows_mut(3, 3).copy_from(&a);
    state_dot
}

fn main() {
    // Initialize EOP
    initialize_eop().unwrap();

    // Create numerical Jacobian for variational equations
    let jacobian = DNumericalJacobian::central(Box::new(dynamics_for_jac))
        .with_fixed_offset(0.1);

    // Configuration for high accuracy
    let config = IntegratorConfig::adaptive(1e-12, 1e-10);

    // Create two integrators:
    // 1. With Jacobian - propagates STM alongside state
    let integrator_nominal = RKN1210DIntegrator::with_config(
        6,
        Box::new(dynamics),
        Some(Box::new(jacobian)),
        None,
        None,
        config.clone(),
    );

    // 2. Without Jacobian - for direct perturbation integration
    let integrator_pert = RKN1210DIntegrator::with_config(
        6,
        Box::new(dynamics),
        None,
        None,
        None,
        config,
    );

    // Initial state: circular orbit at 500 km altitude
    let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
    let state0_static = state_koe_to_eci(oe0, AngleFormat::Degrees);
    let state0 = DVector::from_vec(state0_static.as_slice().to_vec());

    // Apply 10-meter perturbation in X direction
    let perturbation = DVector::from_vec(vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Integration parameters
    let total_time = 100.0; // Total propagation time (seconds)
    let num_steps = 10;     // Number of steps
    let dt = total_time / num_steps as f64;

    // Initialize states
    let mut state_nominal = state0.clone();
    let mut phi = DMatrix::identity(6, 6); // State Transition Matrix starts as identity
    let mut state_pert = &state0 + &perturbation;
    let mut t = 0.0;

    println!("STM vs Direct Perturbation Comparison");
    println!("{}", "=".repeat(70));
    println!("Initial orbit: {:.1} km altitude (circular)", (oe0[0] - R_EARTH) / 1e3);
    println!("Perturbation: {:.1} m in X direction", perturbation[0]);
    println!("Propagating for {:.0} seconds in {} steps\n", total_time, num_steps);
    println!("Theory: For small δx₀, the perturbed state should satisfy:");
    println!("        x_pert(t) ≈ x_nominal(t) + Φ(t,t₀)·δx₀\n");
    println!("Step   Time(s)  ||Error||(m)  Max Component(m)  Relative Error");
    println!("{}", "-".repeat(70));

    for step in 0..num_steps {
        // Propagate nominal trajectory with STM
        let result_nominal = integrator_nominal.step_with_varmat(
            t,
            state_nominal.clone(),
            None,
            phi.clone(),
            Some(dt),
        );
        let new_state_nominal = result_nominal.state;
        let new_phi = result_nominal.phi.unwrap();
        let dt_used = result_nominal.dt_used;

        // Propagate perturbed trajectory directly
        let result_pert = integrator_pert.step(
            t,
            state_pert.clone(),
            None,
            Some(dt),
        );

        // Predict perturbed state using STM: x_pert ≈ x_nominal + Φ·δx₀
        let state_pert_predicted = &new_state_nominal + &new_phi * &perturbation;

        // Compute error between STM prediction and direct integration
        let error = &result_pert.state - &state_pert_predicted;
        let error_norm = error.norm();
        let error_max = error.abs().max();

        // Relative error compared to perturbation magnitude
        let relative_error = error_norm / perturbation.norm();

        println!(
            "{:4}  {:7.1}  {:12.6}  {:16.6}  {:13.6}",
            step + 1,
            t + dt_used,
            error_norm,
            error_max,
            relative_error
        );

        // Update for next step
        state_nominal = new_state_nominal;
        phi = new_phi;
        state_pert = result_pert.state;
        t += dt_used;
    }
}

// Example output:
// STM vs Direct Perturbation Comparison
// ======================================================================
// Initial orbit: 500.0 km altitude (circular)
// Perturbation: 10.0 m in X direction
// Propagating for 100 seconds in 10 steps
//
// Theory: For small δx₀, the perturbed state should satisfy:
//         x_pert(t) ≈ x_nominal(t) + Φ(t,t₀)·δx₀
//
// Step   Time(s)  ||Error||(m)  Max Component(m)  Relative Error
// ----------------------------------------------------------------------
//    1     10.0      0.000078          0.000053      0.000008
//    2     20.0      0.000299          0.000211      0.000030
//    3     30.0      0.000627          0.000462      0.000063
//    4     40.0      0.001025          0.000791      0.000103
//    5     50.0      0.001463          0.001176      0.000146
//    6     60.0      0.001919          0.001600      0.000192
//    7     70.0      0.002378          0.002057      0.000238
//    8     80.0      0.002831          0.002539      0.000283
//    9     90.0      0.003271          0.003040      0.000327
//   10    100.0      0.003693          0.003556      0.000369
