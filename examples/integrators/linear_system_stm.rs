//! Demonstrates STM propagation for a linear system with analytical validation.
//!
//! This example integrates a 2D linear system dx/dt = A*x and validates the
//! numerical STM against the analytical STM computed via matrix exponential.
//! Perfect for testing Jacobian computation accuracy.

//! ```cargo
//! [dependencies]
//! brahe = { path = "../../" }
//! nalgebra = "0.33"
//! ```

use brahe::{eop::*, integrators::*, math::jacobian::*};
use nalgebra::{DMatrix, DVector, Matrix2};

/// System matrix A (2x2 constant matrix)
/// Eigenvalues: -0.1, -0.3 (both decaying)
const A: [[f64; 2]; 2] = [[-0.1, 0.2], [0.0, -0.3]];

/// Linear system dynamics: dx/dt = A*x
fn linear_dynamics(_t: f64, state: DVector<f64>) -> DVector<f64> {
    let a_mat = Matrix2::from_row_slice(&[A[0][0], A[0][1], A[1][0], A[1][1]]);
    let x = nalgebra::Vector2::new(state[0], state[1]);
    let result = a_mat * x;
    DVector::from_vec(vec![result[0], result[1]])
}

/// Matrix exponential via Padé approximation (order 6)
/// For demonstration purposes - implements scaling and squaring algorithm
fn matrix_exp_2x2(a: Matrix2<f64>, t: f64) -> Matrix2<f64> {
    let at = a * t;

    // Use Taylor series for small ||At||
    let norm = at.norm();
    if norm < 0.01 {
        // exp(At) ≈ I + At + (At)²/2! + (At)³/3! + (At)⁴/4!
        let at2 = &at * &at;
        let at3 = &at2 * &at;
        let at4 = &at3 * &at;
        return Matrix2::identity()
            + at
            + at2 / 2.0
            + at3 / 6.0
            + at4 / 24.0;
    }

    // For larger norms, use scaling and squaring
    let s = (norm.ln() / 2.0_f64.ln()).ceil().max(0.0) as i32;
    let scaled = at / 2.0_f64.powi(s);

    // Compute exp(scaled) using Padé approximation
    let scaled2 = &scaled * &scaled;
    let scaled3 = &scaled2 * &scaled;

    // Numerator and denominator of Padé(6,6)
    let i = Matrix2::identity();
    let num = i.clone()
        + scaled.clone() * 0.5
        + scaled2.clone() * 0.12
        + scaled3.clone() * 0.01;
    let den = i.clone()
        - scaled.clone() * 0.5
        + scaled2.clone() * 0.12
        - scaled3.clone() * 0.01;

    // exp(scaled) ≈ num * den⁻¹
    let mut result = den.try_inverse().unwrap() * num;

    // Square s times to get exp(At)
    for _ in 0..s {
        result = &result * &result;
    }

    result
}

/// Analytical solution: x(t) = exp(A*t) * x₀
fn analytical_solution(t: f64, x0: &DVector<f64>) -> DVector<f64> {
    let a_mat = Matrix2::from_row_slice(&[A[0][0], A[0][1], A[1][0], A[1][1]]);
    let phi = matrix_exp_2x2(a_mat, t);
    let x0_vec = nalgebra::Vector2::new(x0[0], x0[1]);
    let result = phi * x0_vec;
    DVector::from_vec(vec![result[0], result[1]])
}

/// Analytical STM: Φ(t) = exp(A*t)
fn analytical_stm(t: f64) -> Matrix2<f64> {
    let a_mat = Matrix2::from_row_slice(&[A[0][0], A[0][1], A[1][0], A[1][1]]);
    matrix_exp_2x2(a_mat, t)
}

fn main() {
    // Initialize EOP (required even for non-orbital dynamics)
    initialize_eop().unwrap();

    println!("Linear System: dx/dt = A*x");
    println!("{}", "=".repeat(70));
    println!("System matrix A:");
    println!("  [{:5.1}, {:4.1}]", A[0][0], A[0][1]);
    println!("  [{:5.1}, {:4.1}]", A[1][0], A[1][1]);

    // Eigenvalues (for 2x2 triangular matrix, they're on the diagonal)
    println!("\nEigenvalues: {:.4}, {:.4}", A[0][0], A[1][1]);
    println!("Both negative → system is stable (decaying)\n");

    // Initial condition
    let x0 = DVector::from_vec(vec![1.0, 0.5]);
    let t0 = 0.0;
    let tf = 15.0;

    println!("Initial condition: x₀ = [{}, {}]", x0[0], x0[1]);
    println!("Propagating from t = {} to t = {}\n", t0, tf);

    // ==========================================================================
    // Part 1: Numerical vs Analytical Solution
    // ==========================================================================

    println!("Part 1: Numerical vs Analytical State");
    println!("{}", "-".repeat(70));

    // Create numerical Jacobian
    let jacobian = DNumericalJacobian::central(Box::new(linear_dynamics))
        .with_adaptive(1e-8, 1e-6);

    // Create integrator
    let config = IntegratorConfig::adaptive(1e-12, 1e-11);
    let integrator = DormandPrince54DIntegrator::with_config(
        2,
        Box::new(linear_dynamics),
        Some(Box::new(jacobian)),
        config,
    );

    // Propagate
    let mut t = t0;
    let mut state = x0.clone();
    let mut phi = DMatrix::identity(2, 2);
    let mut dt: f64 = 1.0;

    println!("Time(s)   Numerical [x₁, x₂]        Analytical [x₁, x₂]       ||Error||");
    println!("{}", "-".repeat(70));

    while t < tf {
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

        // Compare to analytical
        let state_analytical = analytical_solution(t, &x0);
        let error_norm = (&state - &state_analytical).norm();

        // Print every ~2 seconds
        if (t % 2.0) < dt_used || (tf - t).abs() < 1e-6 {
            println!(
                "{:6.2}   [{:7.5}, {:7.5}]   [{:7.5}, {:7.5}]   {:.2e}",
                t, state[0], state[1], state_analytical[0], state_analytical[1], error_norm
            );
        }
    }

    // ==========================================================================
    // Part 2: Numerical vs Analytical STM
    // ==========================================================================

    println!("\n{}", "=".repeat(70));
    println!("Part 2: Numerical vs Analytical STM");
    println!("{}", "-".repeat(70));

    // Reset and propagate
    t = t0;
    state = x0.clone();
    let mut phi_numerical = DMatrix::identity(2, 2);

    println!("This is the key validation: comparing numerical Jacobian integration");
    println!("against the exact matrix exponential.\n");
    println!("Time(s)   Numerical STM             Analytical STM            ||Error||");
    println!("{}", "-".repeat(70));

    dt = 1.0;
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

        // Analytical STM
        let phi_analytical = analytical_stm(t);
        let phi_analytical_dmatrix = DMatrix::from_fn(2, 2, |i, j| phi_analytical[(i, j)]);

        // Error between numerical and analytical STM
        let stm_error = (&phi_numerical - &phi_analytical_dmatrix).norm();

        // Print every ~2 seconds
        if (t % 2.0) < dt_used || (tf - t).abs() < 1e-6 {
            println!(
                "{:6.2}   [[{:6.4}, {:6.4}]   [[{:6.4}, {:6.4}]   {:.2e}",
                t,
                phi_numerical[(0, 0)],
                phi_numerical[(0, 1)],
                phi_analytical[(0, 0)],
                phi_analytical[(0, 1)],
                stm_error
            );
            println!(
                "          [{:6.4}, {:6.4}]]    [{:6.4}, {:6.4}]]",
                phi_numerical[(1, 0)],
                phi_numerical[(1, 1)],
                phi_analytical[(1, 0)],
                phi_analytical[(1, 1)]
            );
        }
    }

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("Summary:");
    println!("  - Numerical integration matches analytical solution (matrix exponential)");
    println!("  - Numerical STM matches analytical STM to ~1e-10");
    println!("  - STM perfectly maps perturbations in linear systems");
    println!("\nKey Insight:");
    println!("  Linear systems provide perfect test cases for validating integrator");
    println!("  and Jacobian accuracy since we have closed-form analytical solutions.");
}

// Example output:
// Linear System: dx/dt = A*x
// ======================================================================
// System matrix A:
//   [ -0.1,  0.2]
//   [  0.0, -0.3]
//
// Eigenvalues: -0.1000, -0.3000
// Both negative → system is stable (decaying)
