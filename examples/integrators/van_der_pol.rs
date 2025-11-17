//! Demonstrates integration of the stiff Van der Pol oscillator.
//!
//! The Van der Pol oscillator is a nonlinear system with limit cycle behavior.
//! For large μ, it becomes stiff, challenging adaptive step control. This example
//! shows how different tolerances affect integration performance and accuracy.

//! ```cargo
//! [dependencies]
//! brahe = { path = "../../" }
//! nalgebra = "0.33"
//! ```

use brahe::{eop::*, integrators::*, math::jacobian::*};
use nalgebra::{DMatrix, DVector};

// Van der Pol parameter (controls stiffness)
const MU: f64 = 5.0;

/// Van der Pol oscillator: ẍ - μ(1 - x²)ẋ + x = 0
/// As a system: ẋ₁ = x₂, ẋ₂ = μ(1 - x₁²)x₂ - x₁
fn van_der_pol(_t: f64, state: DVector<f64>) -> DVector<f64> {
    let x1 = state[0];
    let x2 = state[1];
    let dx1 = x2;
    let dx2 = MU * (1.0 - x1 * x1) * x2 - x1;
    DVector::from_vec(vec![dx1, dx2])
}

fn main() {
    // Initialize EOP (required even for non-orbital dynamics)
    initialize_eop().unwrap();

    // Initial conditions (standard Van der Pol initialization)
    let x0 = 2.0; // Initial position (away from equilibrium)
    let v0 = 0.0; // Initial velocity
    let state0 = DVector::from_vec(vec![x0, v0]);

    // Time parameters
    let t0 = 0.0;
    let tf = 30.0; // Propagate long enough to see limit cycle

    println!("Van der Pol Oscillator: ẍ - μ(1 - x²)ẋ + x = 0");
    println!("{}", "=".repeat(70));
    println!("Stiffness parameter μ = {}", MU);
    if MU < 2.0 {
        println!("  → Mildly nonlinear (easy to integrate)");
    } else if MU < 7.0 {
        println!("  → Moderately stiff (challenges some integrators)");
    } else {
        println!("  → Very stiff (requires tight tolerances)");
    }
    println!("\nInitial conditions: x₀ = {}, v₀ = {}", x0, v0);
    println!("System exhibits limit cycle oscillation\n");

    // ==========================================================================
    // Part 1: Integration with Appropriate Tolerances
    // ==========================================================================

    println!("Part 1: Integration with High Accuracy Tolerances");
    println!("{}", "-".repeat(70));

    // Create numerical Jacobian
    let jacobian = DNumericalJacobian::central(Box::new(van_der_pol))
        .with_adaptive(1e-8, 1e-6);

    // Use tight tolerances for stiff system
    let config = IntegratorConfig::adaptive(1e-9, 1e-8);
    let integrator = DormandPrince54DIntegrator::with_config(
        2,
        Box::new(van_der_pol),
        Some(Box::new(jacobian)),
        config,
    );

    // Propagate and track statistics
    let mut t = t0;
    let mut state = state0.clone();
    let mut phi = DMatrix::identity(2, 2);
    let mut dt: f64 = 0.1;

    let mut total_steps = 0;
    let mut total_time_integrated = 0.0;
    let mut min_dt = f64::INFINITY;
    let mut max_dt: f64 = 0.0;

    println!("Time(s)   x₁        x₂        dt_used    Steps");
    println!("{}", "-".repeat(70));

    let mut positions = vec![state[0]];
    let mut velocities = vec![state[1]];
    let mut times = vec![t];

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

        total_steps += 1;
        total_time_integrated += dt_used;
        min_dt = min_dt.min(dt_used);
        max_dt = max_dt.max(dt_used);

        positions.push(state[0]);
        velocities.push(state[1]);
        times.push(t);

        // Print every ~2 seconds
        if (t % 2.0) < dt_used || (tf - t).abs() < 1e-6 {
            println!(
                "{:7.2}   {:8.5}  {:8.5}   {:9.6}   {:5}",
                t, state[0], state[1], dt_used, total_steps
            );
        }
    }

    println!("\nIntegration Statistics:");
    println!("  Total steps: {}", total_steps);
    println!("  Average step size: {:.6} s", total_time_integrated / total_steps as f64);
    println!("  Min step size: {:.6} s", min_dt);
    println!("  Max step size: {:.6} s", max_dt);
    println!("  Step size ratio (max/min): {:.1}", max_dt / min_dt);

    // ==========================================================================
    // Part 2: Effect of Tolerance on Performance
    // ==========================================================================

    println!("\n{}", "=".repeat(70));
    println!("Part 2: Effect of Tolerance on Integration Performance");
    println!("{}", "-".repeat(70));

    // Test different tolerance levels
    let tolerances = vec![
        (1e-6, 1e-5, "Loose"),
        (1e-9, 1e-8, "Tight"),
        (1e-12, 1e-11, "Very Tight"),
    ];

    println!("{:15}  {:>7}  {:>10}  {:>10}", "Tolerance", "Steps", "Avg dt(s)", "Final x₁");
    println!("{}", "-".repeat(70));

    for (abs_tol, rel_tol, label) in tolerances {
        let jacobian = DNumericalJacobian::central(Box::new(van_der_pol))
            .with_adaptive(1e-8, 1e-6);

        let config = IntegratorConfig::adaptive(abs_tol, rel_tol);
        let integrator = DormandPrince54DIntegrator::with_config(
            2,
            Box::new(van_der_pol),
            Some(Box::new(jacobian)),
            config,
        );

        let mut t = t0;
        let mut state = state0.clone();
        let mut phi = DMatrix::identity(2, 2);
        let mut dt: f64 = 0.1;
        let mut steps = 0;

        while t < tf {
            let (new_state, new_phi, dt_used, _, dt_next) = integrator.step_with_varmat(
                t,
                state.clone(),
                phi.clone(),
                dt.min(tf - t),
            );
            t += dt_used;
            state = new_state;
            phi = new_phi;
            dt = dt_next;
            steps += 1;
        }

        let avg_dt = tf / steps as f64;
        println!("{:15}  {:7}  {:10.6}  {:10.5}", label, steps, avg_dt, state[0]);
    }

    // ==========================================================================
    // Part 3: Phase Space Analysis
    // ==========================================================================

    println!("\n{}", "=".repeat(70));
    println!("Part 3: Phase Space and Limit Cycle");
    println!("{}", "-".repeat(70));

    // Find approximate limit cycle bounds
    let x1_min = positions.iter().copied().fold(f64::INFINITY, f64::min);
    let x1_max = positions.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let x2_min = velocities.iter().copied().fold(f64::INFINITY, f64::min);
    let x2_max = velocities.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    println!("Van der Pol oscillator converges to a limit cycle in phase space");
    println!("\nPhase space bounds after {:.0} seconds:", tf);
    println!("  x₁ range: [{:.4}, {:.4}]", x1_min, x1_max);
    println!("  x₂ range: [{:.4}, {:.4}]", x2_min, x2_max);

    // Estimate period from zero crossings (approximate)
    let mut zero_crossings = Vec::new();
    for i in 1..positions.len() {
        if positions[i - 1] * positions[i] < 0.0 {
            // Linear interpolation to find crossing time
            let t_cross = times[i - 1]
                + (times[i] - times[i - 1]) * positions[i - 1].abs()
                    / (positions[i - 1].abs() + positions[i].abs());
            zero_crossings.push(t_cross);
        }
    }

    if zero_crossings.len() >= 3 {
        // Period is twice the interval between adjacent crossings (up and down)
        let mut periods = Vec::new();
        for i in (0..zero_crossings.len() - 1).step_by(2) {
            periods.push(zero_crossings[i + 1] - zero_crossings[i]);
        }
        let avg_period: f64 = periods.iter().sum::<f64>() / periods.len() as f64;
        println!("\nEstimated limit cycle period: {:.2} s", avg_period);
        println!("Frequency: {:.3} Hz", 1.0 / avg_period);
    }

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("Summary:");
    println!(
        "  - Van der Pol oscillator with μ = {} is {}",
        MU,
        if MU > 5.0 { "stiff" } else { "moderately nonlinear" }
    );
    println!("  - Adaptive step control handles stiffness by reducing step size");
    println!("  - Step size varies by factor of {:.0}x during integration", max_dt / min_dt);
    println!("  - Tighter tolerances require more steps but improve accuracy");
    println!("  - System converges to stable limit cycle oscillation");
    println!("\nKey Insight:");
    println!("  Stiff systems require adaptive integrators with appropriate tolerances.");
    println!("  Fixed-step methods would require very small dt everywhere, wasting compute.");
}

// Example output:
// Van der Pol Oscillator: ẍ - μ(1 - x²)ẋ + x = 0
// ======================================================================
// Stiffness parameter μ = 5
//   → Moderately stiff (challenges some integrators)
//
// Initial conditions: x₀ = 2, v₀ = 0
// System exhibits limit cycle oscillation
