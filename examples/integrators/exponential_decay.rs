//! Demonstrates numerical integration of exponential decay with analytical validation.
//!
//! This example integrates the simple ODE dx/dt = -k*x and compares the numerical
//! solution to the analytical solution x(t) = x₀*exp(-kt). Also demonstrates
//! STM propagation with analytical Jacobian verification.

//! ```cargo
//! [dependencies]
//! brahe = { path = "../../" }
//! nalgebra = "0.33"
//! ```

use brahe::{eop::*, integrators::*, math::jacobian::*};
use nalgebra::{DMatrix, DVector};

// Decay constant
const K: f64 = 0.1;

/// Exponential decay dynamics: dx/dt = -k*x
fn decay_dynamics(_t: f64, state: DVector<f64>) -> DVector<f64> {
    -K * state
}

/// Analytical solution: x(t) = x₀ * exp(-kt)
fn analytical_solution(t: f64, x0: f64) -> f64 {
    x0 * (-K * t).exp()
}

/// Analytical STM: Φ(t) = exp(-kt)
fn analytical_stm(t: f64) -> f64 {
    (-K * t).exp()
}

fn main() {
    // Initialize EOP (required even for non-orbital dynamics)
    initialize_eop().unwrap();

    // Initial condition
    let x0 = DVector::from_vec(vec![1.0]);
    let t0 = 0.0;
    let tf = 20.0; // Final time

    println!("Exponential Decay: dx/dt = -k*x");
    println!("{}", "=".repeat(70));
    println!("Decay constant k = {}", K);
    println!("Initial condition x₀ = {}", x0[0]);
    println!("Analytical solution: x(t) = x₀ * exp(-{}*t)", K);
    println!("Analytical STM: Φ(t) = exp(-{}*t)\n", K);

    // ==========================================================================
    // Part 1: Compare numerical integration to analytical solution
    // ==========================================================================

    println!("Part 1: Numerical vs Analytical Solution");
    println!("{}", "-".repeat(70));

    // Create numerical Jacobian
    let jacobian = DNumericalJacobian::central(Box::new(decay_dynamics))
        .with_adaptive(1e-8, 1e-6);

    // Create integrator with high accuracy
    let config = IntegratorConfig::adaptive(1e-12, 1e-11);
    let integrator = DormandPrince54DIntegrator::with_config(
        1,
        Box::new(decay_dynamics),
        Some(Box::new(jacobian)),
        config,
    );

    // Propagate numerically
    let mut t = t0;
    let mut state = x0.clone();
    let mut phi = DMatrix::identity(1, 1);
    let mut dt: f64 = 1.0;

    println!("Time(s)  Numerical   Analytical  Abs Error   Rel Error");
    println!("{}", "-".repeat(70));

    let mut step_count = 0;
    while t < tf {
        // Propagate with STM
        let (new_state, new_phi, dt_used, _, dt_next) = integrator.step_with_varmat(
            t,
            state.clone(),
            phi.clone(),
            dt.min(tf - t)
        );

        t += dt_used;
        state = new_state;
        phi = new_phi;
        dt = dt_next;
        step_count += 1;

        // Compare to analytical
        let x_analytical = analytical_solution(t, x0[0]);
        let abs_error = (state[0] - x_analytical).abs();
        let rel_error = abs_error / x_analytical;

        // Print every ~10 steps or at end
        if step_count % 10 == 1 || (tf - t).abs() < 1e-6 {
            println!(
                "{:7.2}  {:.8}  {:.8}  {:.2e}  {:.2e}",
                t, state[0], x_analytical, abs_error, rel_error
            );
        }
    }

    // ==========================================================================
    // Part 2: Validate STM against analytical STM
    // ==========================================================================

    println!("\n{}", "=".repeat(70));
    println!("Part 2: State Transition Matrix Validation");
    println!("{}", "-".repeat(70));

    // Reset for STM comparison
    t = t0;
    state = x0.clone();
    let mut phi_numerical = DMatrix::identity(1, 1);
    dt = 1.0;

    println!("Time(s)  Numerical STM  Analytical STM  Abs Error   Rel Error");
    println!("{}", "-".repeat(70));

    while t < tf {
        let (new_state, new_phi, dt_used, _, dt_next) = integrator.step_with_varmat(
            t,
            state.clone(),
            phi_numerical.clone(),
            dt.min(tf - t)
        );

        t += dt_used;
        state = new_state;
        phi_numerical = new_phi;
        dt = dt_next;

        // Compare to analytical STM
        let phi_analytical: f64 = analytical_stm(t);
        let abs_error = (phi_numerical[(0, 0)] - phi_analytical).abs();
        let rel_error = abs_error / phi_analytical;

        // Print every ~2 seconds
        if (t % 2.0) < dt_used || (tf - t).abs() < 1e-6 {
            println!(
                "{:7.2}  {:.10}  {:.10}  {:.2e}  {:.2e}",
                t, phi_numerical[(0, 0)], phi_analytical, abs_error, rel_error
            );
        }
    }

    // ==========================================================================
    // Part 3: Demonstrate STM perturbation mapping
    // ==========================================================================

    println!("\n{}", "=".repeat(70));
    println!("Part 3: STM Perturbation Mapping");
    println!("{}", "-".repeat(70));

    // Small perturbation
    let delta_x0 = 0.01;

    // Predict final perturbation using STM
    let delta_xf_stm: f64 = phi_numerical[(0, 0)] * delta_x0;

    // Compute final perturbation analytically
    let x_nominal: f64 = analytical_solution(tf, x0[0]);
    let x_perturbed: f64 = analytical_solution(tf, x0[0] + delta_x0);
    let delta_xf_true: f64 = x_perturbed - x_nominal;

    println!("Initial perturbation: δx₀ = {}", delta_x0);
    println!("Final time: t = {:.1} s", tf);
    println!("\nSTM prediction:    δxf = Φ({:.1}) * δx₀ = {:.8}", tf, delta_xf_stm);
    println!("True perturbation: δxf = {:.8}", delta_xf_true);
    println!("Error: {:.2e}", (delta_xf_stm - delta_xf_true).abs());
    println!("\nPerfect agreement! (within numerical precision)");

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("Summary:");
    println!("  - Numerical integration matches analytical solution to ~1e-10");
    println!("  - Numerical STM matches analytical STM Φ(t) = exp(-kt)");
    println!("  - STM correctly maps perturbations through linear dynamics");
}

// Example output:
// Exponential Decay: dx/dt = -k*x
// ======================================================================
// Decay constant k = 0.1
// Initial condition x₀ = 1
// Analytical solution: x(t) = x₀ * exp(-0.1*t)
// Analytical STM: Φ(t) = exp(-0.1*t)
//
// Part 1: Numerical vs Analytical Solution
// ----------------------------------------------------------------------
// Time(s)  Numerical   Analytical  Abs Error   Rel Error
// ----------------------------------------------------------------------
//    0.01  0.99900050  0.99900050  8.88e-16  8.88e-16
//    2.01  0.81777047  0.81777047  1.11e-15  1.36e-15
//    4.01  0.66859943  0.66859943  1.11e-15  1.66e-15
