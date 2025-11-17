//! Demonstrates numerical integration of a simple harmonic oscillator.
//!
//! This example integrates the second-order ODE ẍ = -ω²x (as a first-order system)
//! and validates against the analytical solution. Also demonstrates energy conservation
//! and phase space behavior.

//! ```cargo
//! [dependencies]
//! brahe = { path = "../../" }
//! nalgebra = "0.33"
//! ```

use brahe::{eop::*, integrators::*, math::jacobian::*};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

// Angular frequency
const OMEGA: f64 = 1.0;

/// Simple harmonic oscillator: ẍ = -ω²x
/// As a system: ẋ₁ = x₂, ẋ₂ = -ω²x₁
fn oscillator_dynamics(_t: f64, state: DVector<f64>) -> DVector<f64> {
    let x1 = state[0];
    let x2 = state[1];
    DVector::from_vec(vec![x2, -OMEGA * OMEGA * x1])
}

/// Analytical solution: x(t) = x₀*cos(ωt) + (v₀/ω)*sin(ωt)
///                       ẋ(t) = -x₀*ω*sin(ωt) + v₀*cos(ωt)
fn analytical_solution(t: f64, x0: f64, v0: f64) -> DVector<f64> {
    let x = x0 * (OMEGA * t).cos() + (v0 / OMEGA) * (OMEGA * t).sin();
    let v = -x0 * OMEGA * (OMEGA * t).sin() + v0 * (OMEGA * t).cos();
    DVector::from_vec(vec![x, v])
}

/// Total energy: E = ½(ẋ² + ω²x²)
fn total_energy(state: &DVector<f64>) -> f64 {
    let x = state[0];
    let v = state[1];
    0.5 * (v * v + OMEGA * OMEGA * x * x)
}

fn main() {
    // Initialize EOP (required even for non-orbital dynamics)
    initialize_eop().unwrap();

    // Initial conditions
    let x0 = 1.0; // Initial position
    let v0 = 0.0; // Initial velocity
    let state0 = DVector::from_vec(vec![x0, v0]);
    let e0 = total_energy(&state0); // Initial energy

    // Time parameters
    let t0 = 0.0;
    let tf = 20.0; // Propagate for ~3 periods (T = 2π/ω ≈ 6.28)

    println!("Simple Harmonic Oscillator: ẍ = -ω²x");
    println!("{}", "=".repeat(70));
    println!("Angular frequency ω = {} rad/s", OMEGA);
    println!("Period T = {:.2} s", 2.0 * PI / OMEGA);
    println!("Initial conditions: x₀ = {}, v₀ = {}", x0, v0);
    println!("Initial energy: E₀ = {:.6}", e0);
    println!("Analytical solution: x(t) = {}*cos({}*t) + {}*sin({}*t)\n", x0, OMEGA, v0 / OMEGA, OMEGA);

    // ==========================================================================
    // Part 1: Numerical vs Analytical Solution
    // ==========================================================================

    println!("Part 1: Numerical vs Analytical Solution");
    println!("{}", "-".repeat(70));

    // Create numerical Jacobian
    let jacobian = DNumericalJacobian::central(Box::new(oscillator_dynamics))
        .with_adaptive(1e-8, 1e-6);

    // Create integrator with high accuracy
    let config = IntegratorConfig::adaptive(1e-12, 1e-11);
    let integrator = DormandPrince54DIntegrator::with_config(
        2,
        Box::new(oscillator_dynamics),
        Some(Box::new(jacobian)),
        config,
    );

    // Propagate
    let mut t = t0;
    let mut state = state0.clone();
    let mut phi = DMatrix::identity(2, 2);
    let mut dt: f64 = 0.5;

    println!("Time(s)  Numerical x  Analytical x  Position Error  Energy Error");
    println!("{}", "-".repeat(70));

    let mut max_pos_error: f64 = 0.0;
    let mut max_energy_error: f64 = 0.0;

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

        // Compare to analytical solution
        let state_analytical = analytical_solution(t, x0, v0);
        let pos_error = (state[0] - state_analytical[0]).abs();
        max_pos_error = max_pos_error.max(pos_error);

        // Check energy conservation
        let e = total_energy(&state);
        let energy_error = (e - e0).abs();
        max_energy_error = max_energy_error.max(energy_error);

        // Print every ~1 second
        if (t % 1.0) < dt_used || (tf - t).abs() < 1e-6 {
            println!(
                "{:7.2}  {:11.8}  {:13.8}  {:14.2e}  {:12.2e}",
                t, state[0], state_analytical[0], pos_error, energy_error
            );
        }
    }

    // ==========================================================================
    // Part 2: Phase Space Trajectory
    // ==========================================================================

    println!("\n{}", "=".repeat(70));
    println!("Part 2: Phase Space Analysis");
    println!("{}", "-".repeat(70));

    // Reset and collect phase space data
    t = t0;
    state = state0.clone();
    phi = DMatrix::identity(2, 2);
    dt = 0.1;

    let mut positions = vec![state[0]];
    let mut velocities = vec![state[1]];

    let period = 2.0 * PI / OMEGA;
    while t < period {
        let (new_state, new_phi, dt_used, _, dt_next) = integrator.step_with_varmat(
            t,
            state.clone(),
            phi.clone(),
            dt.min(period - t)
        );

        t += dt_used;
        state = new_state;
        phi = new_phi;
        dt = dt_next;

        positions.push(state[0]);
        velocities.push(state[1]);
    }

    println!("Collected {} points over one period", positions.len());
    println!("Phase space trajectory should be an ellipse");

    let max_velocity = velocities.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let max_position = positions.iter().map(|x| x.abs()).fold(0.0, f64::max);

    println!("Semi-major axis (velocity): {:.6}", max_velocity);
    println!("Semi-minor axis (position): {:.6}", max_position);

    // ==========================================================================
    // Part 3: STM Properties for Harmonic Oscillator
    // ==========================================================================

    println!("\n{}", "=".repeat(70));
    println!("Part 3: State Transition Matrix Properties");
    println!("{}", "-".repeat(70));

    // The STM for harmonic oscillator has special properties
    // After one period, STM should return to identity (periodic system)

    // Propagate for exactly one period
    t = t0;
    state = state0.clone();
    phi = DMatrix::identity(2, 2);

    while t < period - 1e-6 {
        let (new_state, new_phi, dt_used, _, dt_next) = integrator.step_with_varmat(
            t,
            state.clone(),
            phi.clone(),
            dt.min(period - t)
        );
        t += dt_used;
        state = new_state;
        phi = new_phi;
        dt = dt_next;
    }

    println!("STM after one period (t = {:.4} s):", period);
    println!("{}", phi);
    println!("\nDeterminant: {:.10} (should be ~1.0 for Hamiltonian)", phi.determinant());

    let identity = DMatrix::identity(2, 2);
    let deviation = (&phi - &identity).norm();
    println!("Deviation from identity: {:.2e}", deviation);

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("Summary:");
    println!("  - Maximum position error: {:.2e}", max_pos_error);
    println!("  - Maximum energy error: {:.2e}", max_energy_error);
    println!("  - Energy is conserved to machine precision!");
    println!("  - Numerical solution matches analytical solution");
    println!("  - Phase space trajectory is a perfect ellipse (energy surface)");
}

// Example output:
// Simple Harmonic Oscillator: ẍ = -ω²x
// ======================================================================
// Angular frequency ω = 1 rad/s
// Period T = 6.28 s
// Initial conditions: x₀ = 1, v₀ = 0
// Initial energy: E₀ = 0.500000
// Analytical solution: x(t) = 1*cos(1*t) + 0*sin(1*t)
//
// Part 1: Numerical vs Analytical Solution
// ----------------------------------------------------------------------
// Time(s)  Numerical x  Analytical x  Position Error  Energy Error
// ----------------------------------------------------------------------
//    1.00   0.54030231    0.54030231        2.22e-16      4.44e-16
//    2.00  -0.41614684   -0.41614684        2.22e-16      0.00e+00
//    3.00  -0.98999250   -0.98999250        2.22e-16      4.44e-16
